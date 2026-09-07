//! 8-connected-component labelling over a dense matrix, ported from
//! `BinaryConnectedComponents` (juicer arrowhead). A cell is foreground when
//! its value is `> threshold`; foreground cells touching by any of the 8
//! neighbours belong to the same component.

use ndarray::Array2;

/// Return the connected components (each a list of `(row, col)` cells) of the
/// cells where `image > threshold`. Equivalent to juicer's 8-point two-pass
/// labelling, re-implemented as a flood fill.
pub(crate) fn detection(image: &Array2<f64>, threshold: f64) -> Vec<Vec<(usize, usize)>> {
    let (r, c) = image.dim();
    let mut visited = vec![vec![false; c]; r];
    let mut components = Vec::new();

    for i in 0..r {
        for j in 0..c {
            if image[[i, j]] > threshold && !visited[i][j] {
                let mut comp = Vec::new();
                let mut stack = vec![(i, j)];
                visited[i][j] = true;
                while let Some((x, y)) = stack.pop() {
                    comp.push((x, y));
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nx = x as i32 + dx;
                            let ny = y as i32 + dy;
                            if nx >= 0 && ny >= 0 && (nx as usize) < r && (ny as usize) < c {
                                let (nx, ny) = (nx as usize, ny as usize);
                                if !visited[nx][ny] && image[[nx, ny]] > threshold {
                                    visited[nx][ny] = true;
                                    stack.push((nx, ny));
                                }
                            }
                        }
                    }
                }
                components.push(comp);
            }
        }
    }
    components
}
