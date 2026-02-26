use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
use kornia_image::Image;

/// Simple Rectangle struct
#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn new(x: i32, y: i32, width: i32, height: i32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(&self) -> i32 {
        self.x + self.width
    }

    pub fn bottom(&self) -> i32 {
        self.y + self.height
    }

    pub fn area(&self) -> i32 {
        self.width * self.height
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 0,
            height: 0,
        }
    }
}

pub fn calculate_iou(a: Rect, b: Rect) -> f64 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = a.right().min(b.right());
    let y2 = a.bottom().min(b.bottom());

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = ((x2 - x1) * (y2 - y1)) as f64;
    let union = a.area() as f64 + b.area() as f64 - intersection;

    if union <= 1e-6 {
        0.0
    } else {
        intersection / union
    }
}

//
// ─────────────────────────────────────────────
// DRAW RECT (HOLLOW)
// ─────────────────────────────────────────────
//

pub fn draw_rect(image: &mut Image<u8, 3>, rect: Rect, color: [u8; 3]) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let data = image.storage.as_mut_slice();
    let thickness = 2;

    let mut set_pixel = |x: i32, y: i32| {
        if x >= 0 && x < width && y >= 0 && y < height {
            let idx = ((y as usize) * (width as usize) + (x as usize)) * 3;
            data[idx] = color[0];
            data[idx + 1] = color[1];
            data[idx + 2] = color[2];
        }
    };

    // Top & Bottom
    for t in 0..thickness {
        for x in rect.x..rect.x + rect.width {
            set_pixel(x, rect.y + t);
            set_pixel(x, rect.y + rect.height - 1 - t);
        }
    }

    // Left & Right
    for t in 0..thickness {
        for y in rect.y..rect.y + rect.height {
            set_pixel(rect.x + t, y);
            set_pixel(rect.x + rect.width - 1 - t, y);
        }
    }
}

//
// ─────────────────────────────────────────────
// DRAW FILLED RECT
// ─────────────────────────────────────────────
//

pub fn draw_filled_rect(image: &mut Image<u8, 3>, rect: Rect, color: [u8; 3]) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let data = image.storage.as_mut_slice();

    let x_start = rect.x.max(0);
    let y_start = rect.y.max(0);
    let x_end = (rect.x + rect.width).min(width);
    let y_end = (rect.y + rect.height).min(height);

    if x_start >= x_end || y_start >= y_end {
        return;
    }

    for y in y_start..y_end {
        for x in x_start..x_end {
            let idx = ((y as usize) * (width as usize) + (x as usize)) * 3;
            data[idx] = color[0];
            data[idx + 1] = color[1];
            data[idx + 2] = color[2];
        }
    }
}

//
// ─────────────────────────────────────────────
// DRAW TEXT
// ─────────────────────────────────────────────
//

pub fn draw_text(
    image: &mut Image<u8, 3>,
    font: &FontRef,
    text: &str,
    x: i32,
    y: i32,
    color: [u8; 3],
) {
    let width = image.width() as i32;
    let height = image.height() as i32;
    let data = image.storage.as_mut_slice();

    let scale = PxScale { x: 20.0, y: 20.0 };
    let scaled_font = font.as_scaled(scale);

    let mut pen_x = x as f32;
    let pen_y = y as f32;

    for c in text.chars() {
        if c.is_control() {
            continue;
        }

        let glyph = scaled_font.scaled_glyph(c);

        if let Some(outlined) = scaled_font.outline_glyph(glyph) {
            let bounds = outlined.px_bounds();

            outlined.draw(|gx, gy, coverage| {
                if coverage <= 0.5 {
                    return;
                }

                let px = bounds.min.x as i32 + gx as i32 + pen_x as i32;
                let py = bounds.min.y as i32 + gy as i32 + pen_y as i32;

                if px >= 0 && px < width && py >= 0 && py < height {
                    let idx =
                        ((py as usize) * (width as usize) + (px as usize)) * 3;

                    data[idx] = color[0];
                    data[idx + 1] = color[1];
                    data[idx + 2] = color[2];
                }
            });
        }

        pen_x += scaled_font.h_advance(scaled_font.glyph_id(c));
    }
}
