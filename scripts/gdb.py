import gdb
from typing import List


class TransparentPrinter(gdb.ValuePrinter):
    def __init__(self, val: gdb.Value, field: str = '__0'):
        self.__val = val
        self.__field = field

    def to_string(self):
        return self.__val[self.__field]


class TransparentGenericPrinter(gdb.ValuePrinter):
    def __init__(self, val: gdb.Value, field: str = '__0'):
        self.__ty = val.type.template_argument(0)
        self.__val = val
        self.__field = field

    def to_string(self):
        return self.__val[self.__field].cast(self.__ty)


class Field:
    def __init__(self, name: str, offset: int, size: int, type: str):
        self.name = name
        self.offset = offset
        self.size = size
        self.type = type

    def convert(self, value: gdb.Value) -> gdb.Value:
        ty = gdb.lookup_type(self.type)

        number = value["__0"]
        mask = (1 << self.size) - 1
        number = (number >> self.offset) & mask

        if self.type in ["i8", "i16", "i32", "i64", "isize"]:
            if number & (1 << (self.size - 1)):
                number = number - (1 << self.size)

        return number.cast(ty)


class BitfieldPrinter(gdb.ValuePrinter):
    def __init__(self, val: gdb.Value, fields: List[Field]):
        self.__fields = fields
        self.__val = val

    def to_string(self):
        return f"{self.__val.type.tag}({self.__val["__0"]})"

    def children(self):
        return [(field.name, field.convert(self.__val)) for field in self.__fields]

    def num_children(self):
        return len(self.__fields)

    def display_hint(self):
        return "structure"


BITFIELDS = {
    "llfree::entry::Tree": [
        Field("free", 0, 13, "usize"),
        Field("huge", 13, 4, "usize"),
        Field("reserved", 17, 1, "bool"),
        Field("movable", 18, 1, "bool"),
    ],
}

def lookup(val: gdb.Value):
    lookup_tag = val.type.tag
    if lookup_tag is None:
        return None
    if lookup_tag.startswith("llfree::atomic::Atom<"):
        return TransparentGenericPrinter(val)
    if lookup_tag.startswith("llfree::util::Align<"):
        return TransparentPrinter(val)
    if lookup_tag.startswith("core::cell::UnsafeCell<"):
        return TransparentPrinter(val, "value")
    if lookup_tag.startswith("spin::mutex::spin::SpinMutex<"):
        return TransparentPrinter(val, "data")

    for tag, fields in BITFIELDS.items():
        if tag == lookup_tag:
            return BitfieldPrinter(val, fields)

    return None


gdb.current_objfile().pretty_printers.append(lookup)
