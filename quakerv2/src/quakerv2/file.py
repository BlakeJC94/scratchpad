import abc
from datetime import datetime


class File(abc.ABC):
    def __init__(self, content: str):
        self.content = content.strip()

    def sorted_content(self, orderby: str) -> str:
        keys_records = [(self.sort_key(r, orderby), r) for r in self.records()]
        records = [r[1] for r in sorted(keys_records, key=lambda x: x[0])]
        return concat_header_records_footer(self.header(), records, self.footer())

    def sort_key(record: str, orderby: str) -> datetime | float:
        pass

    def header(self) -> list[str]:
        return [""]

    def records(self) -> list[str]:
        return [""]

    def footer(self) -> list[str]:
        return [""]


class CsvFile(File):
    def header(self):
        return self.content.split("\n")[:1]

    def records(self):
        return self.content.split("\n")[1:]

    def footer(self):
        return [""]

    def sort_key(record: str, orderby: str) -> datetime | float:
        values = record.split(',')
        if orderby.startswith("time"):
            key = datetime.fromisoformat(values[0].removesuffix("Z"))
        else:
            key = float(record.split(",")[4])
        return key

class TextFile(CsvFile):
    def sort_key(record: str, orderby: str) -> datetime | float:
        values = record.split('|')
        if orderby.startswith("time"):
            key = datetime.fromisoformat(values[1].removesuffix("Z"))
        else:
            key = float(record.split(",")[10])
        return key


FILE_FMTS = {
    "csv": CsvFile,
    "text": TextFile,
}


def get_file(fmt: str, content: str):
    file_fmt = FILE_FMTS[fmt]
    return file_fmt(content)


def join_files(files: list[File]) -> File:
    file_fmt = type(files[0])
    content = concat_header_records_footer(
        files[0].header(),
        sum([f.records() for f in files], []),
        files[-1].footer(),
    )
    return file_fmt(content)

def concat_header_records_footer(header: list[str], records: list[str], footer: list[str]):
    return "\n".join(header + records + footer)
