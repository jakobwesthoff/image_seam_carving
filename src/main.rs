use anyhow::Context;

type Pixel = (f64, f64, f64, f64);

#[derive(Clone)]
struct Matrix<T>
where
    T: Clone,
{
    width: u32,
    height: u32,
    stride: u32,
    data: Vec<T>,
}
impl<T> Matrix<T>
where
    T: Clone + Copy,
    T: Default,
{
    fn new(width: u32, height: u32, stride: u32, data: Vec<T>) -> Self {
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    fn set(&mut self, y: usize, x: usize, value: T) {
        self.data[y * self.stride as usize + x] = value;
    }

    fn get(&self, y: usize, x: usize) -> &T {
        &self.data[y * self.stride as usize + x]
    }

    fn remove(&mut self, y: usize, x: usize) {
        let i = y * self.stride as usize + x;
        self.data
            .copy_within(i + 1..((y * self.stride as usize) + self.width as usize), i);
    }

    fn stride_iter(&self) -> StrideIterator<'_, T> {
        StrideIterator {
            vec: self.data.as_slice(),
            x: 0,
            y: 0,
            stride: self.stride as usize,
            width: self.width as usize,
        }
    }

    fn set_storage_size(&mut self, width: u32, height: u32, stride: u32) {
        self.width = width;
        self.height = height;
        self.stride = stride;
    }

    fn remove_seam(&mut self, seam: &Seam) {
        for y in 0..self.height {
            self.remove(y as usize, seam[y as usize] as usize);
        }
        self.width = self.width - 1;
    }
}

struct StrideIterator<'a, T> {
    vec: &'a [T],
    x: usize,
    y: usize,
    stride: usize,
    width: usize,
}

impl<'a, T> Iterator for StrideIterator<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.y * self.stride + self.x;
        if index >= self.vec.len() {
            return None;
        }

        let value = self.vec.get(index).unwrap();

        self.x += 1;
        if self.x >= self.width {
            self.x = 0;
            self.y += 1;
        }

        Some(value)
    }
}

impl Matrix<Pixel> {
    fn save(&self, path: &str) -> anyhow::Result<()> {
        let image = image::RgbaImage::from_raw(
            self.width,
            self.height,
            self.stride_iter()
                .flat_map(|(r, g, b, a)| {
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
            .stride_iter()
            .cloned()
            .reduce(|acc, value| acc.min(value))
            .unwrap();
        min
    }

    fn max(&self) -> f64 {
        let max = self
            .stride_iter()
            .cloned()
            .reduce(|acc, value| acc.max(value))
            .unwrap();
        max
    }

    fn print_min_max(&self, name: &str) {
        let min = self.min();
        let max = self.max();

        println!("Matrix {} -> min: {}, max: {}", name, min, max);
    }

    fn save(&self, path: &str) -> anyhow::Result<()> {
        let min = self.min();
        let max = self.max();
        let image = image::RgbaImage::from_raw(
            self.width,
            self.height,
            self.stride_iter()
                .map(|value| (value - min) / (max - min)) // Normalize
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

type Seam = Vec<u32>;

fn load_image(path: &str) -> anyhow::Result<Matrix<Pixel>> {
    let image = image::open(path).context("load image")?;

    Ok(Matrix::new(
        image.width(),
        image.height(),
        image.width(),
        image
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
    ))
}

fn calculate_luminance(luminance: &mut Matrix<f64>, image: &Matrix<Pixel>) {
    luminance.set_storage_size(image.width, image.height, image.stride);
    for y in 0..image.height {
        for x in 0..image.width {
            let (r, g, b, _) = image.get(y as usize, x as usize);
            // https://stackoverflow.com/a/596243
            luminance.set(y as usize, x as usize, 0.2126 * r + 0.7152 * g + 0.0722 * b)
        }
    }
}

// https://en.wikipedia.org/wiki/Sobel_operator
// Maybe checkout Prewitt Operator or Laplacian of Gaussian (LoG) as
// alternatives to caluclate the gradient.
fn calculate_gradient(gradient: &mut Matrix<f64>, luminance: &Matrix<f64>) {
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

    gradient.set_storage_size(luminance.width, luminance.height, luminance.stride);

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
            // sqrt calculation as usually done in sobel operator is not needed
            // here, as we only need the size relation of the gradient values
            // between each other
            gradient.set(cy as usize, cx as usize, gx.powi(2) + gy.powi(2));
        }
    }
}

fn calculate_dp(dp: &mut Matrix<f64>, gradient: &Matrix<f64>) {
    dp.set_storage_size(gradient.width, gradient.height, gradient.stride);

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
}

fn calculate_seam(dp: &Matrix<f64>) -> Seam {
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

struct SeamCarver {
    image: Matrix<Pixel>,
    luminance: Matrix<f64>,
    luminance_initialized: bool,
    gradient: Matrix<f64>,
    dp: Matrix<f64>,
}

impl SeamCarver {
    fn with_image(image_path: &str) -> anyhow::Result<Self> {
        let image = load_image(image_path)?;
        Ok(Self {
            luminance: Matrix::new(
                image.width,
                image.height,
                image.stride,
                vec![0.0; image.width as usize * image.height as usize],
            ),
            luminance_initialized: false,
            gradient: Matrix::new(
                image.width,
                image.height,
                image.stride,
                vec![0.0; image.width as usize * image.height as usize],
            ),
            dp: Matrix::new(
                image.width,
                image.height,
                image.stride,
                vec![0.0; image.width as usize * image.height as usize],
            ),
            image,
        })
    }

    fn carve_vertical(&mut self) {
        if !self.luminance_initialized {
            calculate_luminance(&mut self.luminance, &self.image);
            self.luminance_initialized = true
        }
        calculate_gradient(&mut self.gradient, &self.luminance);
        calculate_dp(&mut self.dp, &self.gradient);
        let seam = calculate_seam(&self.dp);
        self.image.remove_seam(&seam);
        self.luminance.remove_seam(&seam);
    }

    fn save_image(&self, file_path: &str) -> anyhow::Result<()> {
        self.image.save(file_path)?;
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let mut carver = SeamCarver::with_image("./assets/Broadway_tower_edit.jpg")?;

    for i in 0..510 {
        println!("Calculate and remove vertical seam: {}", i);
        carver.carve_vertical();
    }

    carver.save_image("./resized.png")?;

    Ok(())
}
