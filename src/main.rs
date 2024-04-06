use anyhow::Context;

struct Image {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Image {
    fn save(&self, path: &str) -> anyhow::Result<()> {
        let image = image::RgbaImage::from_raw(self.width, self.height, self.pixels.clone())
            .context("create image")?;

        image.save(path).context("save image to file")?;

        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let image =
        image::open("./assets/Broadway_tower_edit.jpg").context("load original image")?;

    let image = Image {
        width: image.width(),
        height: image.height(),
        pixels: image.to_rgba8().into_vec(),
    };

    image.save("./output.png").context("save image")?;

    Ok(())
}
