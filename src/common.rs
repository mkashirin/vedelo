use crate::tracker::Track;
use anyhow::Result;
use opencv::{
    core::{AlgorithmHint, Point, Rect, Scalar},
    imgproc,
    prelude::*,
};

pub fn calculate_iou(a: Rect, b: Rect) -> f64 {
    let intersection = a & b;
    let i_area = intersection.area() as f64;
    let u_area = (a.area() + b.area()) as f64 - i_area;

    if u_area <= 1e-6 { 0.0 } else { i_area / u_area }
}

pub fn draw_prediction(frame: &mut Mat, obj: &Track) -> Result<()> {
    let id_val = obj.id as f64;
    let r = (id_val * 123.4) % 255.0;
    let g = (id_val * 234.5) % 255.0;
    let b = (id_val * 345.6) % 255.0;
    let color = Scalar::new(b, g, r, 0.0);

    let rect = obj.last_rect;
    imgproc::rectangle(frame, rect, color, 2, imgproc::LINE_8, 0)?;

    let label = format!("ID:{} C:{}", obj.id, obj.class_id);
    let mut baseline = 0;
    let text_size = imgproc::get_text_size(
        &label,
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        1,
        &mut baseline,
    )?;

    let label_bg = Rect::new(
        rect.x,
        rect.y - text_size.height - 5,
        text_size.width,
        text_size.height + 5,
    );
    imgproc::rectangle(frame, label_bg, color, -1, imgproc::LINE_8, 0)?;

    imgproc::put_text(
        frame,
        &label,
        Point::new(rect.x, rect.y - 5),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.6,
        Scalar::new(255.0, 255.0, 255.0, 0.0),
        1,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(())
}

#[cfg(target_os = "macos")]
pub fn cvt_color_wrapper(
    src: &Mat,
    dst: &mut Mat,
    code: i32,
    dst_cn: i32,
) -> opencv::Result<()> {
    // MacOS/Newer OpenCV requires 5 arguments.
    imgproc::cvt_color(src, dst, code, dst_cn, AlgorithmHint::ALGO_HINT_DEFAULT)
}

#[cfg(target_os = "linux")]
pub fn cvt_color_wrapper(
    src: &Mat,
    dst: &mut Mat,
    code: i32,
    dst_cn: i32,
) -> opencv::Result<()> {
    // Linux/Older OpenCV requires 4 arguments.
    imgproc::cvt_color(src, dst, code, dst_cn)
}
