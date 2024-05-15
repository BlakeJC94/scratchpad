from argparse import ArgumentParser
from dataclasses import fields

from quakerv2.client import Client
from quakerv2.query import Query


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("argfile", nargs="?")

    for field in fields(Query):
        parser.add_argument(f"--{field.name}", type=field.type, default=None)
    return parser.parse_args()


def main():
    args = vars(parse_args())

    if (args_file := args.pop("args_file", None)) is not None:
        args = {}
        with open(args_file, "r") as f:
            for line in f.readlines():
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k not in ["starttime", "endtime", "updaterafter"]:
                    v = float(v)
                args[k] = v

    client = Client()
    result = client.execute(**args)

    print(result)


if __name__ == "__main__":
    main()
