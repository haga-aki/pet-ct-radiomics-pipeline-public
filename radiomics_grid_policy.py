"""Feature-source policy for mixed-grid PET radiomics output."""

PET_NONSHAPE_FEATURE_CLASSES = (
    "firstorder",
    "glcm",
    "glrlm",
    "glszm",
    "gldm",
    "ngtdm",
)


def merge_pet_and_shape_results(pet_result, shape_result):
    """Merge native-PET non-shape values with CT-grid shape values.

    Diagnostics remain those of the native PET extraction. Any shape value
    accidentally present in ``pet_result`` is discarded.
    """
    merged = {
        key: value
        for key, value in pet_result.items()
        if not key.startswith("original_shape_")
    }
    merged.update(
        {
            key: value
            for key, value in shape_result.items()
            if key.startswith("original_shape_")
        }
    )
    return merged
