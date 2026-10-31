//! Parallel computation of a pre-order outboard.
//!
//! The bao tree over [`IROH_BLOCK_SIZE`] leaves is the blake3 tree cut off at
//! the block level, so every aligned power-of-two run of blocks is a complete
//! subtree, and in a pre-order outboard the parent nodes of a subtree are
//! stored contiguously. That lets us split the data into pieces of
//! `piece_blocks` blocks, hash each piece on the rayon pool with a single
//! positioned read and a single positioned write for its parents, and then
//! compute the few nodes above the pieces at the end.
use std::{
    cell::RefCell,
    io,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Mutex,
    },
};

use bao_tree::{
    blake3,
    io::sync::{ReadAt, WriteAt},
};
use rayon::prelude::*;

use crate::store::IROH_BLOCK_SIZE;

/// Blocks per piece: 64 * 16 KiB = 1 MiB per task.
pub(super) const PIECE_BLOCKS: u64 = 64;

const BLOCK_BYTES: u64 = IROH_BLOCK_SIZE.bytes() as u64;

/// Computes the root hash of `size` bytes from `data`, writing the pre-order
/// outboard to `outboard` and adding the number of hashed bytes to `progress`
/// as it goes. Stops with [`io::ErrorKind::Interrupted`] once `cancel` is set.
pub(super) fn compute<R, W>(
    data: &R,
    size: u64,
    outboard: W,
    piece_blocks: u64,
    progress: &AtomicU64,
    cancel: &AtomicBool,
) -> io::Result<blake3::Hash>
where
    R: ReadAt + Sync + ?Sized,
    W: WriteAt + Send,
{
    debug_assert!(piece_blocks.is_power_of_two());
    let blocks = blocks(size);
    let outboard = Mutex::new(outboard);
    if blocks <= piece_blocks {
        let hash = piece(data, 0, size, 0, true, &outboard)?;
        progress.fetch_add(size, Ordering::Relaxed);
        return Ok(hash);
    }

    let mut pieces = Vec::new();
    layout(0, blocks, 0, piece_blocks, &mut pieces);
    let hashes = pieces
        .par_iter()
        .map(|&(start_block, node)| {
            if cancel.load(Ordering::Relaxed) {
                return Err(io::ErrorKind::Interrupted.into());
            }
            let start = start_block * BLOCK_BYTES;
            let end = size.min(start + piece_blocks * BLOCK_BYTES);
            let hash = piece(data, start, end, node, false, &outboard)?;
            progress.fetch_add(end - start, Ordering::Relaxed);
            Ok(hash)
        })
        .collect::<io::Result<Vec<_>>>()?;

    // The nodes above the pieces, written in contiguous runs.
    let mut top = Vec::new();
    let root = combine(0, blocks, 0, piece_blocks, &hashes, true, &mut top);
    let mut outboard = outboard.into_inner().unwrap();
    let mut buf = Vec::new();
    let mut run_start = 0;
    for (node, (l, r)) in top {
        if !buf.is_empty() && node * 64 != run_start + buf.len() as u64 {
            outboard.write_all_at(run_start, &buf)?;
            buf.clear();
        }
        if buf.is_empty() {
            run_start = node * 64;
        }
        buf.extend_from_slice(l.as_bytes());
        buf.extend_from_slice(r.as_bytes());
    }
    if !buf.is_empty() {
        outboard.write_all_at(run_start, &buf)?;
    }
    outboard.flush()?;
    Ok(root)
}

/// Number of leaves of a blob of `size` bytes (an empty blob has one).
fn blocks(size: u64) -> u64 {
    size.div_ceil(BLOCK_BYTES).max(1)
}

/// Blocks in the left subtree of a node over `blocks` blocks (`blocks > 1`):
/// the largest power of two strictly smaller than `blocks`, as in blake3.
fn left_blocks(blocks: u64) -> u64 {
    debug_assert!(blocks > 1);
    1 << (63 - (blocks - 1).leading_zeros())
}

/// Walks the tree above the pieces, recording each piece as
/// `(start_block, pre-order index of its first parent)`. A subtree over `n`
/// blocks has `n - 1` parents, so the right child of a node over `n` blocks
/// starts `left_blocks(n)` parents after the node itself.
fn layout(start: u64, blocks: u64, node: u64, piece_blocks: u64, pieces: &mut Vec<(u64, u64)>) {
    if blocks <= piece_blocks {
        pieces.push((start, node));
        return;
    }
    let left = left_blocks(blocks);
    layout(start, left, node + 1, piece_blocks, pieces);
    layout(
        start + left,
        blocks - left,
        node + left,
        piece_blocks,
        pieces,
    );
}

/// Hashes the tree above the pieces from their `hashes`, appending each node
/// as `(pre-order index, (left, right))` to `top`, in pre-order.
fn combine(
    start: u64,
    blocks: u64,
    node: u64,
    piece_blocks: u64,
    hashes: &[blake3::Hash],
    is_root: bool,
    top: &mut Vec<(u64, (blake3::Hash, blake3::Hash))>,
) -> blake3::Hash {
    if blocks <= piece_blocks {
        return hashes[(start / piece_blocks) as usize];
    }
    let left = left_blocks(blocks);
    let slot = top.len();
    top.push((
        node,
        (
            blake3::Hash::from_bytes([0; 32]),
            blake3::Hash::from_bytes([0; 32]),
        ),
    ));
    let l = combine(start, left, node + 1, piece_blocks, hashes, false, top);
    let r = combine(
        start + left,
        blocks - left,
        node + left,
        piece_blocks,
        hashes,
        false,
        top,
    );
    top[slot].1 = (l, r);
    parent_cv(&l, &r, is_root)
}

/// Hashes bytes `start..end`, a complete subtree whose first parent has
/// pre-order index `node`, and writes its parents with one positioned write.
fn piece<R, W>(
    data: &R,
    start: u64,
    end: u64,
    node: u64,
    is_root: bool,
    outboard: &Mutex<W>,
) -> io::Result<blake3::Hash>
where
    R: ReadAt + ?Sized,
    W: WriteAt,
{
    thread_local! {
        // Reused across pieces, since a fresh zeroed allocation per piece
        // costs about as much as hashing it.
        static BUF: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }
    let len = (end - start) as usize;
    let mut parents = Vec::with_capacity(((blocks(end - start) - 1) * 64) as usize);
    let hash = BUF.with_borrow_mut(|buf| {
        if buf.len() < len {
            buf.resize(len, 0);
        }
        let buf = &mut buf[..len];
        data.read_exact_at(start, buf)?;
        io::Result::Ok(subtree(buf, start / 1024, is_root, &mut parents))
    })?;
    if !parents.is_empty() {
        outboard.lock().unwrap().write_all_at(node * 64, &parents)?;
    }
    Ok(hash)
}

/// Hashes an in-memory subtree starting at chunk `start_chunk`, appending its
/// parent pairs to `parents` in pre-order.
fn subtree(data: &[u8], start_chunk: u64, is_root: bool, parents: &mut Vec<u8>) -> blake3::Hash {
    let blocks = blocks(data.len() as u64);
    if blocks == 1 {
        return hash_leaf(start_chunk, data, is_root);
    }
    let split = (left_blocks(blocks) * BLOCK_BYTES) as usize;
    let slot = parents.len();
    parents.extend_from_slice(&[0u8; 64]);
    let l = subtree(&data[..split], start_chunk, false, parents);
    let r = subtree(
        &data[split..],
        start_chunk + split as u64 / 1024,
        false,
        parents,
    );
    parents[slot..slot + 32].copy_from_slice(l.as_bytes());
    parents[slot + 32..slot + 64].copy_from_slice(r.as_bytes());
    parent_cv(&l, &r, is_root)
}

fn hash_leaf(start_chunk: u64, data: &[u8], is_root: bool) -> blake3::Hash {
    use blake3::hazmat::{ChainingValue, HasherExt};
    if is_root {
        debug_assert!(start_chunk == 0);
        blake3::hash(data)
    } else {
        let mut hasher = blake3::Hasher::new();
        hasher.set_input_offset(start_chunk * 1024);
        hasher.update(data);
        let non_root_hash: ChainingValue = hasher.finalize_non_root();
        blake3::Hash::from(non_root_hash)
    }
}

fn parent_cv(left: &blake3::Hash, right: &blake3::Hash, is_root: bool) -> blake3::Hash {
    use blake3::hazmat::{merge_subtrees_non_root, merge_subtrees_root, Mode};
    let left = *left.as_bytes();
    let right = *right.as_bytes();
    if is_root {
        merge_subtrees_root(&left, &right, Mode::Hash)
    } else {
        blake3::Hash::from(merge_subtrees_non_root(&left, &right, Mode::Hash))
    }
}

#[cfg(test)]
mod tests {
    use bao_tree::io::outboard::PreOrderMemOutboard;

    use super::*;
    use crate::store::fs::meta::raw_outboard_size;

    fn test_data(size: usize) -> Vec<u8> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..size)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    #[test]
    fn matches_serial_outboard() {
        const K: usize = 1024;
        const B: usize = 16 * K;
        let sizes = [
            0,
            1,
            K - 1,
            K,
            K + 1,
            B - 1,
            B,
            B + 1,
            2 * B,
            2 * B + 1,
            3 * B + 7,
            5 * B,
            7 * B + 1,
            8 * B,
            9 * B - 1,
            64 * B,
            64 * B + 1,
            100 * B + 123,
            129 * B,
            300 * B + 5,
        ];
        for size in sizes {
            let data = test_data(size);
            let expected = PreOrderMemOutboard::create(&data, IROH_BLOCK_SIZE);
            assert_eq!(
                expected.data.len() as u64,
                raw_outboard_size(size as u64),
                "size {size}"
            );
            for piece_blocks in [1, 2, 4, 64] {
                let mut outboard = Vec::new();
                let progress = AtomicU64::new(0);
                let root = compute(
                    &data.as_slice(),
                    size as u64,
                    &mut outboard,
                    piece_blocks,
                    &progress,
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(
                    root, expected.root,
                    "root, size {size}, piece {piece_blocks}"
                );
                assert_eq!(root, blake3::hash(&data), "blake3, size {size}");
                assert_eq!(
                    outboard, expected.data,
                    "outboard, size {size}, piece {piece_blocks}"
                );
                assert_eq!(progress.load(Ordering::Relaxed), size as u64);
            }
        }
    }
}
