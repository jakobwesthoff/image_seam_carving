use anyhow::Context;

type Pixel = (f64, f64, f64, f64);
struct Matrix<T> {
    width: u32,
    height: u32,
    data: Vec<T>,
}
impl<T> Matrix<T> {
    fn set(&mut self, y: usize, x: usize, value: T) {
        self.data[y * self.width as usize + x] = value;
    }

    fn get(&self, y: usize, x: usize) -> &T {
        &self.data[y * self.width as usize + x]
    }
}

impl Matrix<Pixel> {
    fn save(&self, path: &str) -> anyhow::Result<()> {
        let image = image::RgbaImage::from_raw(
            self.width,
            self.height,
            self.data
                .iter()
                .flat_map(|(r, b, g, a)| {
                    vec![
                        (r * 255.0).round().clamp(0.0, 255.0) as u8,
                        (g * 255.0).round().clamp(0.0, 255.0) as u8,
                        (b * 255.0).round().clamp(0.0, 255.0) as u8,
                        (a * 255.0).round().clamp(0.0, 255.0) as u8,
                    ]
                })
                .collect(),
        )
        .context("create image")?;

        image.save(path).context("save image to file")?;

        Ok(())
    }
}

impl Matrix<f64> {
    fn min(&self) -> f64 {
        let min = self
            .data
            .iter()
            .cloned()
            .reduce(|acc, value| acc.min(value))
            .unwrap();
        min
    }

    fn max(&self) -> f64 {
        let max = self
            .data
            .iter()
            .cloned()
            .reduce(|acc, value| acc.max(value))
            .unwrap();
        max
    }

    fn normalize(&self) -> Self {
        let max = self.max();
        Self {
            width: self.width,
            height: self.height,
            data: self.data.iter().map(|value| value / max).collect(),
        }
    }

    fn save(&self, path: &str) -> anyhow::Result<()> {
        let normalized_image = self.normalize();
        let image = image::RgbaImage::from_raw(
            self.width,
            self.height,
            normalized_image
                .data
                .iter()
                .flat_map(|value| {
                    vec![
                        (value * 255.0).round().clamp(0.0, 255.0) as u8,
                        (value * 255.0).round().clamp(0.0, 255.0) as u8,
                        (value * 255.0).round().clamp(0.0, 255.0) as u8,
                        255_u8,
                    ]
                })
                .collect(),
        )
        .context("create image")?;

        image.save(path).context("save image to file")?;

        Ok(())
    }
}

type ImageMatrix = Matrix<Pixel>;
type LuminanceMatrix = Matrix<f64>;
type GradientMatrix = Matrix<f64>;

fn load_image(path: &str) -> anyhow::Result<ImageMatrix> {
    let image = image::open(path).context("load image")?;

    Ok(Matrix {
        width: image.width(),
        height: image.height(),
        data: image
            .to_rgba8()
            .into_vec()
            .chunks(4)
            .map(|pixel| {
                (
                    pixel[0] as f64 / 255.0,
                    pixel[1] as f64 / 255.0,
                    pixel[2] as f64 / 255.0,
                    pixel[3] as f64 / 255.0,
                )
            })
            .collect(),
    })
}

fn calculate_luminance(image: &ImageMatrix) -> LuminanceMatrix {
    Matrix {
        width: image.width,
        height: image.height,
        data: image
            .data
            .iter()
            // .map(|(r, g, b, _a)| 0.2126 * r + 0.7152 * g + 0.0722 * b)
            .map(|(r, g, b, _a)| 0.299 * r + 0.587 * g + 0.114 * b)
            // .map(|(r, g, b, _a)| (0.299 * r.powi(2) + 0.587 * g.powi(2) + 0.114 * b.powi(2)).sqrt())
            .collect(),
    }
}

// https://en.wikipedia.org/wiki/Sobel_operator
// Maybe checkout Prewitt Operator or Laplacian of Gaussian (LoG) as
// alternatives to caluclate the gradient.
fn calculate_gradient(luminance: &LuminanceMatrix) -> GradientMatrix {
    #[rustfmt::skip]
    static G_X: [[f64; 3]; 3] = [
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0]
    ];

    #[rustfmt::skip]
    static G_Y: [[f64; 3]; 3] = [
        [ 1.0,  2.0,  1.0],
        [ 0.0,  0.0,  0.0],
        [-1.0, -2.0, -1.0]
    ];

    let mut gradient = GradientMatrix {
        width: luminance.width,
        height: luminance.height,
        data: vec![0.0; luminance.width as usize * luminance.height as usize],
    };

    for cy in 0..luminance.height as isize {
        for cx in 0..luminance.width as isize {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x < 0
                        || x >= luminance.width as isize
                        || y < 0
                        || y >= luminance.height as isize
                    {
                        // Skip out of bounds
                        continue;
                    }
                    gx += G_X[(dy + 1) as usize][(dx + 1) as usize]
                        * luminance.get(y as usize, x as usize);
                    gy += G_Y[(dy + 1) as usize][(dx + 1) as usize]
                        * luminance.get(y as usize, x as usize);
                }
            }
            gradient.set(cy as usize, cx as usize, (gx.powi(2) + gy.powi(2)).sqrt());
        }
    }
    gradient
}

fn main() -> anyhow::Result<()> {
    let image = load_image("./assets/Broadway_tower_edit.jpg")?;

    // Calculate luminance of the image
    // https://stackoverflow.com/a/596243
    let luminance = calculate_luminance(&image);
    luminance
        .save("./luminance.png")
        .context("save luminance")?;

    // Apply sobel operator to get luminance gradient
    let gradient = calculate_gradient(&luminance);
    gradient.save("./gradient.png").context("save gradient")?;

    Ok(())
}
