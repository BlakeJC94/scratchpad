from argparse import ArgumentParser

from quakerv2.client import Client


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("argfile", nargs="?")

    parser.add_argument("--endtime", default=None)
    parser.add_argument("--starttime", default=None)
    parser.add_argument("--updatedafter", default=None)

    parser.add_argument("--mindepth", default=None)
    parser.add_argument("--maxdepth", default=None)
    parser.add_argument("--minmagnitude", default=None)
    parser.add_argument("--maxmagnitude", default=None)

    parser.add_argument("--minlatitude", default=None)
    parser.add_argument("--maxlatitude", default=None)
    parser.add_argument("--minlongitude", default=None)
    parser.add_argument("--maxlongitude", default=None)

    parser.add_argument("--latitude", default=None)
    parser.add_argument("--longitude", default=None)
    parser.add_argument("--maxradius", default=None)
    parser.add_argument("--maxradiuskm", default=None)

    return parser.parse_args()

def main():
    args = vars(parse_args())

    if (args_file := args.pop("args_file", None)) is not None:
        with open(args_file, 'r') as f:
            args = {}
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
