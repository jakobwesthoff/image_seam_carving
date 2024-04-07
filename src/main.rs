use anyhow::Context;

type Pixel = Option<(f64, f64, f64, f64)>;

#[derive(Clone)]
struct Matrix<T>
where
    T: Clone,
{
    width: u32,
    height: u32,
    data: Vec<T>,
}
impl<T> Matrix<T>
where
    T: Clone,
{
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
                .filter(|pixel| pixel.is_some())
                .flat_map(|pixel| {
                    let (r, g, b, a) = pixel.unwrap();
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

    fn remove_seam(&mut self, seam: &Seam) {
        'iterate_rows: for y in 0..self.height {
            // Find x by scanning the row, as we are storing the image with Optional values.
            // This is not the most intelligent method and more or less a hack,
            // but was the quickest and dirtiest way to make it work I could
            // think of in the middle of the night :).
            let mut index = 0;
            let stride = self.data.len() / self.height as usize;
            for x in 0..stride {
                if self.data[y as usize * stride + x].is_some() {
                    if index == seam[y as usize] {
                        self.data[y as usize * stride + x as usize] = None;
                        continue 'iterate_rows;
                    } else {
                        index += 1;
                    }
                }
            }
        }
        self.width = self.width - 1;
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
        let min = self.min();
        let max = self.max();
        Self {
            width: self.width,
            height: self.height,
            data: self
                .data
                .iter()
                .map(|value| (value - min) / (max - min))
                .collect(),
        }
    }

    fn print_min_max(&self, name: &str) {
        let min = self.min();
        let max = self.max();

        println!("Matrix {} -> min: {}, max: {}", name, min, max);
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
type DpMatrix = Matrix<f64>;
type Seam = Vec<u32>;

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
                Some((
                    pixel[0] as f64 / 255.0,
                    pixel[1] as f64 / 255.0,
                    pixel[2] as f64 / 255.0,
                    pixel[3] as f64 / 255.0,
                ))
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
            .filter(|pixel| pixel.is_some())
            .map(|pixel| {
                let (r, g, b, _) = pixel.unwrap();
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            // .map(|(r, g, b, _a)| 0.299 * r + 0.587 * g + 0.114 * b)
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

fn calculate_dp(gradient: &GradientMatrix) -> DpMatrix {
    let mut dp = DpMatrix {
        width: gradient.width,
        height: gradient.height,
        data: vec![0.0; gradient.width as usize * gradient.height as usize],
    };

    // Start with the initial row as it is
    for x in 0..gradient.width as usize {
        dp.set(0, x, *gradient.get(0, x));
    }

    // Each next energy value is the value itself plus the minimum from the
    // three pixels above it
    for y in 1..gradient.height {
        for cx in 0..gradient.width as isize {
            let mut min_energy_on_path = f64::MAX;
            for dx in -1..=1 {
                let x = cx + dx;
                if x < 0 || x >= gradient.width as isize {
                    continue;
                }
                min_energy_on_path = min_energy_on_path.min(*dp.get(y as usize - 1, x as usize));
            }
            dp.set(
                y as usize,
                cx as usize,
                gradient.get(y as usize, cx as usize) + min_energy_on_path,
            );
        }
    }

    dp
}

fn calculate_seam(dp: &DpMatrix) -> Seam {
    // Find minimum energy path start point
    let mut min_x = f64::MAX;
    let mut min_x_index = 0;
    for x in 0..dp.width {
        let candidate_min_x = dp.get(dp.height as usize - 1, x as usize);
        if *candidate_min_x < min_x {
            min_x_index = x;
            min_x = *candidate_min_x;
        }
    }

    let mut seam = vec![0; dp.height as usize];
    seam[dp.height as usize - 1] = min_x_index;

    for y in (0..=dp.height - 2).rev() {
        min_x = f64::MAX;
        let mut next_min_x_index = min_x_index;
        for dx in -1..=1 as isize {
            let x = min_x_index as isize + dx;
            if x < 0 || x >= dp.width as isize {
                continue;
            }
            let candidate_min_x = dp.get(y as usize, x as usize);
            if *candidate_min_x < min_x {
                next_min_x_index = x as u32;
                min_x = *candidate_min_x;
            }
        }
        min_x_index = next_min_x_index;
        seam[y as usize] = min_x_index;
    }

    seam
}

fn main() -> anyhow::Result<()> {
    let mut image = load_image("./assets/Broadway_tower_edit.jpg")?;

    for i in 0..510 {
        println!("Calculate and remove vertical seam: {}", i);
        // Calculate luminance of the image
        // https://stackoverflow.com/a/596243
        let luminance = calculate_luminance(&image);
        // luminance.print_min_max("luminance");
        // luminance
        //     .save("./luminance.png")
        //     .context("save luminance")?;

        // Apply sobel operator to get luminance gradient
        let gradient = calculate_gradient(&luminance);
        // gradient.print_min_max("gradient");
        // gradient.save("./gradient.png").context("save gradient")?;

        // Use Dynamic Programming to calculate the prerequisite to find the path
        // with the lowest energy.
        let dp = calculate_dp(&gradient);
        // dp.print_min_max("dp");
        // dp.save("dp.png")?;

        let seam = calculate_seam(&dp);

        // let mut new_image = image.clone();
        // for y in 0..seam.len() {
        //     new_image.set(y, seam[y] as usize, (255.0, 255.0, 0.0, 255.0));
        // }

        // new_image.save("./seam.png")?;

        image.remove_seam(&seam);
    }

    image.save("./resized.png")?;

    Ok(())
}
