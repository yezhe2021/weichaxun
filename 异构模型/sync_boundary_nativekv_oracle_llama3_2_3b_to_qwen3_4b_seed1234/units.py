from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ByteToken:
    index: int
    start: int
    end: int


@dataclass(frozen=True)
class SyncUnit:
    unit_id: int
    byte_start: int
    byte_end: int
    char_start: int
    char_end: int
    raw_text: str
    llama_indices: tuple
    qwen_indices: tuple
    llama_right: int
    qwen_right: int

    def json(self):
        value = asdict(self)
        value["llama_indices"] = list(self.llama_indices)
        value["qwen_indices"] = list(self.qwen_indices)
        return value


def char_byte_boundaries(text):
    result = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8")); result.append(total)
    return result


def byte_tokens(spans, char_to_byte):
    return [ByteToken(span.index, char_to_byte[span.start], char_to_byte[span.end])
            for span in spans if span.index != 0 and not span.special]


def build_synchronized_units(text, llama_spans, qwen_spans):
    char_to_byte = char_byte_boundaries(text)
    byte_to_char = {value: index for index, value in enumerate(char_to_byte)}
    llama, qwen = byte_tokens(llama_spans, char_to_byte), byte_tokens(qwen_spans, char_to_byte)
    shared = sorted(({token.end for token in llama} & {token.end for token in qwen}) | {0})
    units = []
    for left, right in zip(shared, shared[1:]):
        li = tuple(token.index for token in llama if left < token.end <= right)
        qi = tuple(token.index for token in qwen if left < token.end <= right)
        lr = [token.index for token in llama if token.end == right]
        qr = [token.index for token in qwen if token.end == right]
        if not li or not qi or not lr or not qr: continue
        cs, ce = byte_to_char[left], byte_to_char[right]
        units.append(SyncUnit(len(units), left, right, cs, ce, text[cs:ce], li, qi, max(lr), max(qr)))
    if not units: raise RuntimeError("No synchronized units")
    return units


def region_bounds(total_bytes, regions):
    return [(math.floor(i * total_bytes / regions), math.floor((i + 1) * total_bytes / regions))
            for i in range(regions)]


def overlap(unit, bounds): return max(0, min(unit.byte_end, bounds[1]) - max(unit.byte_start, bounds[0]))


def unit_score(unit, importance, family, reduction):
    indices = unit.llama_indices if family == "llama" else unit.qwen_indices
    values = importance[list(indices)]
    return float(values.max() if reduction == "max" else values.sum())


def select_units(units, importance, family, reduction, regions, unique):
    total_bytes = max(unit.byte_end for unit in units); chosen = []; used = set()
    for bounds in region_bounds(total_bytes, regions):
        candidates = [unit for unit in units if overlap(unit, bounds) > 0]
        if not candidates: candidates = units
        ranked = sorted(candidates, key=lambda unit: (unit_score(unit, importance, family, reduction),
                                                       overlap(unit, bounds), -unit.unit_id), reverse=True)
        selected = next((unit for unit in ranked if not unique or unit.unit_id not in used), None)
        if selected is None:
            center = (bounds[0] + bounds[1]) / 2
            unused = [unit for unit in units if unit.unit_id not in used]
            if unused:
                selected = min(unused, key=lambda unit: (abs((unit.byte_start + unit.byte_end) / 2 - center), unit.unit_id))
            else:
                # A fixed 32-entry receiver budget cannot be uniquely filled when the
                # entire sample has fewer than 32 synchronized units. Preserve every
                # unique unit, then use a deterministic region-best repeat as padding.
                selected = ranked[0]
        chosen.append(selected); used.add(selected.unit_id)
    return chosen


def trigger_units(units, anchor_characters):
    result = []
    for anchor in anchor_characters:
        containing = [unit for unit in units if unit.char_start <= anchor < unit.char_end]
        if containing: result.append(containing[0]); continue
        result.append(min(units, key=lambda unit: (min(abs(anchor - unit.char_start), abs(anchor - unit.char_end)), unit.unit_id)))
    return result
