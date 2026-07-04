use crate::app_settings::AppSettings;
use crate::exif_processing::{read_exposure_time_secs, read_iso};
use crate::formats::is_raw_file;
use crate::image_loader::load_base_image_from_bytes;
use crate::image_processing::{
    apply_cpu_default_raw_processing, apply_linear_to_srgb, apply_srgb_to_linear,
};
use crate::panorama_stitching::{Feature, KeyPoint};
use crate::panorama_utils::{processing, stitching};
use image::{DynamicImage, GenericImageView, GrayImage, Rgb32FImage};
use nalgebra::{Matrix3, Point2};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tauri::{AppHandle, Emitter};

pub type HdrFrame = (String, DynamicImage, Duration, f32);

const DEGHOST_FAST_THRESHOLD: u8 = 8;
const DEGHOST_NON_MAXIMA_SUPPRESSION_RADIUS: f32 = 8.0;

struct FrameDetection {
    keypoints: Vec<KeyPoint>,
    features: Vec<Feature>,
    scale_factor: f64,
}

pub fn load_hdr_frames(
    paths: &[String],
    app_handle: &AppHandle,
    settings: &AppSettings,
) -> Result<Vec<HdrFrame>, String> {
    assert!(paths.len() >= 2, "hdr merge requires at least two paths");
    paths
        .iter()
        .map(|path| {
            let _ = app_handle.emit(
                "hdr-progress",
                format!(
                    "Processing '{}'",
                    Path::new(path)
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                ),
            );
            let file_bytes =
                fs::read(path).map_err(|e| format!("Failed to read image {}: {}", path, e))?;
            let mut dynamic_image =
                load_base_image_from_bytes(&file_bytes, path, false, settings, None)
                    .map_err(|e| format!("Failed to load image {}: {}", path, e))?;
            if !is_raw_file(path) {
                dynamic_image = apply_srgb_to_linear(dynamic_image);
            }
            let gains = match read_iso(path, &file_bytes) {
                None => return Err(format!("Image {} is missing ISO/Sensitivity data", path)),
                Some(gains) => gains as f32,
            };
            let exposure = match read_exposure_time_secs(path, &file_bytes) {
                None => return Err(format!("Image {} is missing ExposureTime data", path)),
                Some(exp) => Duration::from_secs_f32(exp),
            };
            Ok((path.clone(), dynamic_image, exposure, gains))
        })
        .collect()
}

pub fn assert_uniform_dimensions(frames: &[HdrFrame]) -> Result<(), String> {
    assert!(!frames.is_empty(), "dimension check requires at least one frame");
    let (first_path, first_image, _, _) = &frames[0];
    let width = first_image.width();
    let height = first_image.height();
    for (path, image, _, _) in frames.iter().skip(1) {
        if image.width() != width || image.height() != height {
            return Err(format!(
                "Dimension mismatch detected.\n\nBase image ({}): {}x{}\nTarget image ({}): {}x{}\n\nHDR merge requires all images to be exactly the same size.",
                Path::new(first_path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                width,
                height,
                Path::new(path)
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                image.width(),
                image.height()
            ));
        }
    }
    Ok(())
}

pub fn align_hdr_frames(frames: &mut [HdrFrame], app_handle: &AppHandle) {
    assert!(!frames.is_empty(), "alignment requires at least one frame");
    let _ = app_handle.emit("hdr-progress", "Deghosting...");
    let brief_pairs = processing::generate_brief_pairs();
    let reference_index = frames.len() / 2;
    let detections: Vec<FrameDetection> = frames
        .iter()
        .map(|frame| {
            let label = Path::new(&frame.0)
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            detect_frame_features(&frame.1, &brief_pairs, &label, is_raw_file(&frame.0))
        })
        .collect();
    for (index, detection) in detections.iter().enumerate() {
        println!(
            "[deghost] frame '{}': {} features (reference={})",
            frames[index].0,
            detection.features.len(),
            index == reference_index
        );
    }
    for index in 0..frames.len() {
        if index == reference_index {
            continue;
        }
        let file_name = Path::new(&frames[index].0)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned();
        let _ = app_handle.emit("hdr-progress", format!("Aligning '{}'...", file_name));
        let aligned = align_frame_to_reference(
            &frames[index].1,
            &detections[index],
            &detections[reference_index],
        );
        match aligned {
            Some(warped) => frames[index].1 = DynamicImage::ImageRgb32F(warped),
            None => {
                let _ = app_handle.emit(
                    "hdr-progress",
                    format!("Could not align '{}', using as-is", file_name),
                );
            }
        }
    }
}

fn detect_frame_features(
    image: &DynamicImage,
    brief_pairs: &[(Point2<i32>, Point2<i32>)],
    debug_label: &str,
    source_is_raw: bool,
) -> FrameDetection {
    let mut detection_proxy = image.clone();
    if source_is_raw {
        apply_cpu_default_raw_processing(&mut detection_proxy);
    } else {
        detection_proxy = apply_linear_to_srgb(detection_proxy);
    }
    let gray_full = image::imageops::colorops::grayscale(&detection_proxy.to_rgb8());
    let (width, height) = gray_full.dimensions();
    let (small_width, small_height, scale_factor) =
        processing::calculate_downscale_dimensions(width, height);
    let gray_small = image::imageops::resize(
        &gray_full,
        small_width,
        small_height,
        image::imageops::FilterType::Triangle,
    );
    let normalized = processing::normalize_grayscale(&gray_small);
    debug_dump_normalized(debug_label, &normalized);
    let features = processing::find_features_tuned(
        &normalized,
        brief_pairs,
        DEGHOST_FAST_THRESHOLD,
        DEGHOST_NON_MAXIMA_SUPPRESSION_RADIUS,
    );
    let keypoints = features.iter().map(|feature| feature.keypoint).collect();
    FrameDetection {
        keypoints,
        features,
        scale_factor,
    }
}

fn debug_dump_normalized(label: &str, normalized: &GrayImage) {
    let path = std::env::temp_dir().join(format!("rapidraw_deghost_{}.png", label));
    match normalized.save(&path) {
        Ok(()) => println!("[deghost] normalized image written to {}", path.display()),
        Err(e) => println!("[deghost] failed to write normalized image for '{}': {}", label, e),
    }
}

fn align_frame_to_reference(
    frame_image: &DynamicImage,
    frame: &FrameDetection,
    reference: &FrameDetection,
) -> Option<Rgb32FImage> {
    let matches = processing::match_features(&reference.features, &frame.features);
    println!(
        "[deghost] matches against reference: {} (threshold {})",
        matches.len(),
        processing::MIN_INLIERS_FOR_CONNECTION
    );
    if matches.len() < processing::MIN_INLIERS_FOR_CONNECTION {
        return None;
    }
    let (_, inliers) =
        match processing::find_homography_ransac(&matches, &reference.keypoints, &frame.keypoints) {
            Some(result) => result,
            None => {
                println!("[deghost] RANSAC found too few inliers");
                return None;
            }
        };
    println!("[deghost] inliers: {}", inliers.len());
    let inlier_points: Vec<(Point2<f64>, Point2<f64>)> = inliers
        .iter()
        .map(|m| {
            let reference_point = reference.keypoints[m.index1];
            let frame_point = frame.keypoints[m.index2];
            (
                Point2::new(reference_point.x as f64, reference_point.y as f64),
                Point2::new(frame_point.x as f64, frame_point.y as f64),
            )
        })
        .collect();
    let homography_small = processing::compute_homography(&inlier_points)?;
    let reference_scale_inverse = scaling_matrix(1.0 / reference.scale_factor);
    let frame_scale = scaling_matrix(frame.scale_factor);
    let homography_full = frame_scale * homography_small * reference_scale_inverse;
    let source = frame_image.to_rgb32f();
    let (width, height) = frame_image.dimensions();
    Some(stitching::warp_image_homography(
        &source,
        &homography_full,
        width,
        height,
    ))
}

fn scaling_matrix(scale: f64) -> Matrix3<f64> {
    Matrix3::new(scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0)
}

#[cfg(test)]
mod align_hdr_frames_tests {
    use super::align_frame_to_reference;
    use crate::panorama_utils::processing::generate_brief_pairs;
    use crate::panorama_utils::stitching::warp_image_homography;
    use image::{DynamicImage, GenericImageView, Rgb32FImage};
    use nalgebra::Matrix3;

    fn textured_frame() -> DynamicImage {
        let mut img = Rgb32FImage::new(320, 320);
        for y in 0..320u32 {
            for x in 0..320u32 {
                let mut hash = x.wrapping_mul(374761393).wrapping_add(y.wrapping_mul(668265263));
                hash = (hash ^ (hash >> 13)).wrapping_mul(1274126177);
                let value = (hash & 0xff) as f32 / 255.0;
                img.put_pixel(x, y, image::Rgb([value, value, value]));
            }
        }
        DynamicImage::ImageRgb32F(img)
    }

    fn detect(image: &DynamicImage) -> super::FrameDetection {
        super::detect_frame_features(image, &generate_brief_pairs(), "test", false)
    }

    #[test]
    fn realigns_translated_frame_to_reference() {
        let reference_image = textured_frame();
        let shift = Matrix3::new(1.0, 0.0, 3.0, 0.0, 1.0, 2.0, 0.0, 0.0, 1.0);
        let shifted = warp_image_homography(&reference_image.to_rgb32f(), &shift, 320, 320);
        let shifted_image = DynamicImage::ImageRgb32F(shifted);

        let reference = detect(&reference_image);
        let frame = detect(&shifted_image);
        let aligned = align_frame_to_reference(&shifted_image, &frame, &reference)
            .expect("alignment should succeed on textured frame");

        let reference_pixels = reference_image.to_rgb32f();
        let mut error = 0.0f32;
        for y in 80..240u32 {
            for x in 80..240u32 {
                error += (aligned.get_pixel(x, y)[0] - reference_pixels.get_pixel(x, y)[0]).abs();
            }
        }
        let mean_error = error / (160.0 * 160.0);
        assert!(mean_error < 0.1, "mean realignment error too high: {}", mean_error);
    }

    #[test]
    fn returns_none_on_featureless_frame() {
        let flat = DynamicImage::ImageRgb32F(Rgb32FImage::from_pixel(
            160,
            160,
            image::Rgb([0.5, 0.5, 0.5]),
        ));
        let reference = detect(&textured_frame());
        let frame = detect(&flat);
        assert!(align_frame_to_reference(&flat, &frame, &reference).is_none());
    }
}
