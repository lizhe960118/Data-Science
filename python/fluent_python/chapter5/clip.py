def clip(text, max_len = 10):
    end = None
    if len(text) > max_len:
        space_before = text.rfind(' ', 0, max_len)
        if space_before >= 0:
            end = space_before
        else:
            space_after = text.rfind(' ', max_len)
            if space_after >= 0:
                end = space_after
    if end is None:
        end = len(text)

    return text[:end].rstrip()

print(clip('banana ', 6))
print(clip('banana ', 7))
print(clip('banana ', 5))
print(clip('banana split', 6))
print(clip('banana split', 7))
print(clip('banana split', 10))
print(clip('banana split', 11))
print(clip('banana split', 12))
