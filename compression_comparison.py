import zlib
import zstandard as zstd

# A: read the compressed file from baseline run
with open("final_model.int8.ptz", "rb") as f:
    zlib_blob = f.read()

zlib_size = len(zlib_blob)
print(f"Current zlib compressed size: {zlib_size:,} bytes ({zlib_size / 1_000_000:.2f} MB)")


# B: read the compressed file from the new run
raw_bytes = zlib.decompress(zlib_blob)
raw_size = len(raw_bytes)
print(f"Raw uncompressed size: {raw_size:,} bytes ({raw_size / 1_000_000:.2f} MB)")

# C recompress with zstd at max compression
compressor = zstd.ZstdCompressor(level=22)
zstd_blob = compressor.compress(raw_bytes)
zstd_size = len(zstd_blob)
print(f"Zstd compressed size: {zstd_size:,} bytes ({zstd_size / 1_000_000:.2f} MB)")

# D compare
savings = zlib_size - zstd_size
print(f"\nSavings: {savings:,} bytes ({savings / 1_000:.1f} KB)")
print(f"That's {savings} extra int8 parameters you could fit")

# E sweep different zstd levels
print("\n--- zstd lvel sweep ---")
for level in [1, 3, 6, 10, 15, 19, 22]:
    compressor = zstd.ZstdCompressor(level=level)
    blob = compressor.compress(raw_bytes)
    print(f"  level {level:2d}: {len(blob):,} bytes  (savings: {zlib_size - len(blob):,})")
