import abc


class File(abc.ABC):
    def __init__(self, content: str):
        self.content = content.strip()

    def header(self) -> list[str]:
        return []

    def records(self) -> list[str]:
        return []

    def footer(self) -> list[str]:
        return []


class CsvFile(File):
    def header(self):
        return self.content.split("\n")[:1]

    def records(self):
        return self.content.split("\n")[1:]

    def footer(self):
        return []


class TextFile(CsvFile):
    pass


class GeojsonFile(File):
    def header(self):
        lines = self.content.split("\n")
        return [lines[0].split("[", 1)[0] + "["]

    def footer(self):
        lines = self.content.split("\n")
        return ["]" + "]".join(lines[-1].split("]")[2:]).removesuffix(",")]

    def records(self):
        lines = self.content.split("\n")
        return [
            lines[0].split("[", 1)[1],
            *[l for l in lines[1:-1]],
            "]".join(lines[-1].split("]")[:2]).removesuffix(","),
        ]


class KmlFile(File):
    def header(self):
        return self.content.split("\n")[:13]

    def records(self):
        return self.content.split("\n")[13:-3]

    def footer(self):
        return self.content.split("\n")[-3:]


class XmlFile(File):
    def header(self):
        return self.content.split("\n")[:3]

    def records(self):
        return self.content.split("\n")[3:-2]

    def footer(self):
        return self.content.split("\n")[-2:]


class QuakemlFile(XmlFile):
    pass


FILE_FMTS = {
    "csv": CsvFile,
    "text": TextFile,
    "geojson": GeojsonFile,
    "kml": KmlFile,
    "xml": XmlFile,
    "quakeml": QuakemlFile,
    None: GeojsonFile,
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
