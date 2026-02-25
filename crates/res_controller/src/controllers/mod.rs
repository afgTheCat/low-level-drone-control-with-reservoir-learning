pub mod esn;
pub mod izhikevich_controller;

use nalgebra::DMatrix;

// horizontally stack two matricies
pub fn hstack(mut left: DMatrix<f64>, right: DMatrix<f64>) -> DMatrix<f64> {
    if right.ncols() == 0 {
        return left;
    }
    assert_eq!(left.nrows(), right.nrows(), "hstack: row count mismatch");
    let mut out = DMatrix::zeros(left.nrows(), left.ncols() + right.ncols());
    let (r, c_left) = (left.nrows(), left.ncols());
    out.view_mut((0, 0), (r, c_left)).copy_from(&left);
    out.view_mut((0, c_left), (r, right.ncols()))
        .copy_from(&right);
    left = out;
    left
}
