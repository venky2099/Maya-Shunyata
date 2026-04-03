# sign_paper.py — IP protection for Maya-Shunyata (Paper 8)
# LSB steganographic signature in matplotlib figures
# Canary: MayaNexusVS2026NLL_Bengaluru_Narasimha

import numpy as np
import struct

SIGNATURE = (
    "MayaNexusVS2026NLL_Bengaluru_Narasimha | "
    "ORCID:0000-0002-3315-7907 | "
    "Nexus Learning Labs Bengaluru | "
    "Maya-Shunyata Paper 8"
)


def sign_figure(img_array: np.ndarray) -> np.ndarray:
    signed = img_array.copy().astype(np.uint8)
    flat   = signed.flatten()
    payload = SIGNATURE.encode("utf-8")
    length  = len(payload)
    header  = struct.pack(">I", length)
    bits    = []
    for byte in header + payload:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    if len(bits) > len(flat):
        raise ValueError("Image too small to sign.")
    for idx, bit in enumerate(bits):
        flat[idx] = (flat[idx] & 0xFE) | bit
    return flat.reshape(signed.shape)


def save_signed_figure(fig, path: str) -> None:
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img        = Image.open(buf).convert("RGB")
    arr        = np.array(img)
    signed_arr = sign_figure(arr)
    signed_img = Image.fromarray(signed_arr)
    signed_img.save(path)
    print(f"[sign_paper] Signed figure saved: {path}")


if __name__ == "__main__":
    print(f"Signature: {SIGNATURE}")
    print("sign_paper.py ready — Maya-Shunyata Paper 8.")
