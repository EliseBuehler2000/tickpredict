import CoreGraphics
import CoreVideo
import Foundation
import ImageIO

/// The preprocessing recipe recorded in each Core ML model's metadata.
nonisolated struct PreprocessingSpec: Sendable, Decodable, Equatable {
    let imageSize: Int
    let bottomCropFraction: Double
    let padColor: [UInt8]

    enum CodingKeys: String, CodingKey {
        case imageSize = "image_size"
        case bottomCropFraction = "bottom_crop_fraction"
        case padColor = "pad_color"
    }
}

nonisolated enum ImagePreparationError: Error {
    case unreadableImage
    case imageTooSmall
    case pixelBufferUnavailable
}

/// Tightly packed 8-bit RGB pixels, row major.
nonisolated struct RGBImage: Sendable {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(pixels.count == width * height * 3)
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    init(width: Int, height: Int, fill: [UInt8]) {
        self.width = width
        self.height = height
        pixels = [UInt8](repeating: 0, count: width * height * 3)
        for i in stride(from: 0, to: pixels.count, by: 3) {
            pixels[i] = fill[0]; pixels[i + 1] = fill[1]; pixels[i + 2] = fill[2]
        }
    }
}

/// Reproduces the Python training pipeline (Pillow 13) exactly:
/// EXIF transpose -> RGB -> remove bottom strip -> `ImageOps.pad` with Lanczos.
///
/// Pixel values are read without colour management, as Pillow does, so the
/// model sees the same numbers it was validated on.
nonisolated enum ImagePreparation {
    /// Decode image data and apply its EXIF orientation.
    static func decode(_ data: Data) throws -> RGBImage {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ImagePreparationError.unreadableImage }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1
        return try rawRGB(image, orientation: orientation)
    }

    /// Full pipeline for one model's input.
    static func prepare(_ image: RGBImage, spec: PreprocessingSpec) throws -> RGBImage {
        guard min(image.width, image.height) >= 2 else { throw ImagePreparationError.imageTooSmall }
        let keptRows = Int(pythonRound(Double(image.height) * (1 - spec.bottomCropFraction)))
        return pad(image, rows: max(keptRows, 1), size: spec.imageSize, color: spec.padColor)
    }

    // MARK: - Decoding

    /// Draws into an RGB context in the image's own colour space, so no conversion happens,
    /// then packs to RGB and applies the EXIF orientation in the same pass.
    private static func rawRGB(_ image: CGImage, orientation: Int) throws -> RGBImage {
        let width = image.width, height = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil }
            ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw ImagePreparationError.unreadableImage }
        let swaps = (5 ... 8).contains(orientation)
        let outW = swaps ? height : width, outH = swaps ? width : height
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        rgba.withUnsafeBufferPointer { src in
            rgb.withUnsafeMutableBufferPointer { dst in
                parallelChunks(outH) { rows in
                    for y in rows {
                        for x in 0 ..< outW {
                            let (sx, sy) = sourcePixel(x, y, width: width, height: height, orientation: orientation)
                            let from = (sy * width + sx) * 4, to = (y * outW + x) * 3
                            dst[to] = src[from]; dst[to + 1] = src[from + 1]; dst[to + 2] = src[from + 2]
                        }
                    }
                }
            }
        }
        return RGBImage(width: outW, height: outH, pixels: rgb)
    }

    /// Source pixel for output pixel (x, y), matching `ImageOps.exif_transpose` for orientations 1–8.
    @inline(__always)
    private static func sourcePixel(_ x: Int, _ y: Int, width w: Int, height h: Int, orientation: Int) -> (Int, Int) {
        switch orientation {
        case 2: (w - 1 - x, y)            // mirror horizontal
        case 3: (w - 1 - x, h - 1 - y)    // rotate 180
        case 4: (x, h - 1 - y)            // mirror vertical
        case 5: (y, x)                    // transpose
        case 6: (y, h - 1 - x)            // rotate 90 clockwise
        case 7: (w - 1 - y, h - 1 - x)    // transverse
        case 8: (w - 1 - y, x)            // rotate 90 counter-clockwise
        default: (x, y)
        }
    }

    // MARK: - ImageOps.pad / contain

    /// Letterboxes the top `rows` rows of `image` (the rest is the removed bottom strip).
    static func pad(_ image: RGBImage, rows: Int, size: Int, color: [UInt8]) -> RGBImage {
        var width = size, height = size
        let imageRatio = Double(image.width) / Double(rows)
        if imageRatio != 1 {
            if imageRatio > 1 {
                height = max(1, Int(pythonRound(Double(rows) / Double(image.width) * Double(size))))
            } else {
                width = max(1, Int(pythonRound(Double(image.width) / Double(rows) * Double(size))))
            }
        }
        let resized = LanczosResampler.resize(image, rows: rows, width: width, height: height)
        if width == size && height == size { return resized }
        var out = RGBImage(width: size, height: size, fill: color)
        let x0 = width != size ? Int(pythonRound(Double(size - width) * 0.5)) : 0
        let y0 = width != size ? 0 : Int(pythonRound(Double(size - height) * 0.5))
        for y in 0 ..< height {
            let src = y * width * 3
            let dst = ((y + y0) * size + x0) * 3
            out.pixels.replaceSubrange(dst ..< dst + width * 3, with: resized.pixels[src ..< src + width * 3])
        }
        return out
    }

    /// Python 3 `round`: halves go to the even neighbour.
    static func pythonRound(_ value: Double) -> Double {
        value.rounded(.toNearestOrEven)
    }

    // MARK: - Core ML input

    static func pixelBuffer(_ image: RGBImage) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height,
                                  kCVPixelFormatType_32BGRA, attributes, &buffer) == kCVReturnSuccess,
              let buffer
        else { throw ImagePreparationError.pixelBufferUnavailable }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        for y in 0 ..< image.height {
            let row = base + y * rowBytes
            for x in 0 ..< image.width {
                let src = (y * image.width + x) * 3
                row[x * 4] = image.pixels[src + 2]
                row[x * 4 + 1] = image.pixels[src + 1]
                row[x * 4 + 2] = image.pixels[src]
                row[x * 4 + 3] = 255
            }
        }
        return buffer
    }
}

/// Runs `body` over `count` items split into contiguous chunks on all cores.
nonisolated func parallelChunks(_ count: Int, _ body: (Range<Int>) -> Void) {
    let chunks = min(count, ProcessInfo.processInfo.activeProcessorCount * 4)
    guard chunks > 1 else { body(0 ..< count); return }
    // Each chunk writes a disjoint range of rows, so sharing `body` across threads is safe.
    withoutActuallyEscaping(body) { body in
        nonisolated(unsafe) let body = body
        DispatchQueue.concurrentPerform(iterations: chunks) { chunk in
            body(count * chunk / chunks ..< count * (chunk + 1) / chunks)
        }
    }
}

/// A port of Pillow's 8-bit separable resampler (`libImaging/Resample.c`) with
/// the Lanczos (a = 3) filter, including its fixed-point rounding.
nonisolated enum LanczosResampler {
    private static let precisionBits = 32 - 8 - 2
    private static let support = 3.0

    private static func sinc(_ x: Double) -> Double {
        if x == 0 { return 1 }
        let px = x * Double.pi
        return sin(px) / px
    }

    private static func lanczos(_ x: Double) -> Double {
        (-3.0 <= x && x < 3.0) ? sinc(x) * sinc(x / 3) : 0
    }

    private struct Coefficients {
        let kernelSize: Int
        let bounds: [(start: Int, count: Int)]
        let weights: [Int]
    }

    private static func coefficients(inSize: Int, outSize: Int) -> Coefficients {
        let scale = Double(inSize) / Double(outSize)
        let filterScale = max(scale, 1)
        let reach = support * filterScale
        let kernelSize = Int(reach.rounded(.up)) * 2 + 1
        var bounds: [(Int, Int)] = []
        bounds.reserveCapacity(outSize)
        var weights = [Int](repeating: 0, count: outSize * kernelSize)
        var raw = [Double](repeating: 0, count: kernelSize)
        for xx in 0 ..< outSize {
            let center = (Double(xx) + 0.5) * scale
            let inverse = 1 / filterScale
            let xmin = max(Int(center - reach + 0.5), 0)
            let xmax = min(Int(center + reach + 0.5), inSize) - xmin
            var total = 0.0
            for x in 0 ..< xmax {
                let w = lanczos((Double(x + xmin) - center + 0.5) * inverse)
                raw[x] = w
                total += w
            }
            for x in 0 ..< xmax {
                let w = total != 0 ? raw[x] / total : raw[x]
                let scaled = w * Double(1 << precisionBits)
                weights[xx * kernelSize + x] = Int(w < 0 ? -0.5 + scaled : 0.5 + scaled)
            }
            bounds.append((xmin, xmax))
        }
        return Coefficients(kernelSize: kernelSize, bounds: bounds, weights: weights)
    }

    @inline(__always) private static func clip8(_ value: Int) -> UInt8 {
        if value >= (1 << (precisionBits + 8)) { return 255 }
        if value <= 0 { return 0 }
        return UInt8(value >> precisionBits)
    }

    /// Resizes the top `rows` rows of `image`.
    static func resize(_ image: RGBImage, rows: Int, width: Int, height: Int) -> RGBImage {
        var current = width != image.width
            ? horizontal(image, rows: rows, width: width)
            : RGBImage(width: image.width, height: rows, pixels: Array(image.pixels[0 ..< image.width * rows * 3]))
        if height != rows { current = vertical(current, height: height) }
        return current
    }

    private static func horizontal(_ image: RGBImage, rows: Int, width: Int) -> RGBImage {
        let c = coefficients(inSize: image.width, outSize: width)
        let half = 1 << (precisionBits - 1)
        var out = [UInt8](repeating: 0, count: width * rows * 3)
        image.pixels.withUnsafeBufferPointer { src in
            c.weights.withUnsafeBufferPointer { k in
                c.bounds.withUnsafeBufferPointer { bounds in
                    out.withUnsafeMutableBufferPointer { dst in
                        parallelChunks(rows) { chunk in
                            for y in chunk {
                                let row = y * image.width * 3
                                for xx in 0 ..< width {
                                    let (start, count) = bounds[xx]
                                    let kk = xx * c.kernelSize
                                    var r = half, g = half, b = half
                                    for x in 0 ..< count {
                                        let p = row + (start + x) * 3
                                        let w = k[kk + x]
                                        r += Int(src[p]) * w; g += Int(src[p + 1]) * w; b += Int(src[p + 2]) * w
                                    }
                                    let o = (y * width + xx) * 3
                                    dst[o] = clip8(r); dst[o + 1] = clip8(g); dst[o + 2] = clip8(b)
                                }
                            }
                        }
                    }
                }
            }
        }
        return RGBImage(width: width, height: rows, pixels: out)
    }

    private static func vertical(_ image: RGBImage, height: Int) -> RGBImage {
        let c = coefficients(inSize: image.height, outSize: height)
        let half = 1 << (precisionBits - 1)
        let stride = image.width * 3
        var out = [UInt8](repeating: 0, count: image.width * height * 3)
        image.pixels.withUnsafeBufferPointer { src in
            c.weights.withUnsafeBufferPointer { k in
                c.bounds.withUnsafeBufferPointer { bounds in
                    out.withUnsafeMutableBufferPointer { dst in
                        parallelChunks(height) { chunk in
                            for yy in chunk {
                                let (start, count) = bounds[yy]
                                let kk = yy * c.kernelSize
                                for x in 0 ..< image.width {
                                    var r = half, g = half, b = half
                                    for y in 0 ..< count {
                                        let p = (start + y) * stride + x * 3
                                        let w = k[kk + y]
                                        r += Int(src[p]) * w; g += Int(src[p + 1]) * w; b += Int(src[p + 2]) * w
                                    }
                                    let o = (yy * image.width + x) * 3
                                    dst[o] = clip8(r); dst[o + 1] = clip8(g); dst[o + 2] = clip8(b)
                                }
                            }
                        }
                    }
                }
            }
        }
        return RGBImage(width: image.width, height: height, pixels: out)
    }
}
