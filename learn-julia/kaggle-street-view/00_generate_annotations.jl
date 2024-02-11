# Open `data/raw/list_English_Img.m`

using CSV
using DataFrames

function main()
    raw = read_raw_data()
    processed = convert_to_map(raw)
    write_annotations(processed)
    validate_csvs()
end

function read_raw_data()
    raw_annotations_fp = "./data/raw/list_English_Img.m"

    f = open(raw_annotations_fp, "r")
    raw = read(f, String)
    close(f)

    return raw
end

function convert_to_map(raw)
    raw = replace(raw, r"(\d+);(\w)" => s"[\1];\n\2")
    raw = replace(raw, r";" => "")
    raw = replace(raw, r" " => ",")

    foo = Dict()
    for raw_split in split(raw, "]")
        m = match(r".*list\.(\w+).=", raw_split)
        if m === nothing
            println("[nope]")
            continue
        end
        header = m[1]
        println(header)

        raw_split = strip(raw_split)
        raw_split = split(raw_split, "[")[2]
        foo[header] = raw_split * "\n"
    end

    return foo
end

function write_annotations(processed)
    annotations_dir = "./data/annotations"
    for (header, content) in processed
        fp = annotations_dir * "/" * header * ".csv"
        write(fp, content)
    end
end

function validate_csvs()
    annotations_dir = "./data/annotations"
    foo = Dict()
    for i in readdir(annotations_dir)
        if i[end-2:end] != "csv"
            continue
        end
        println("Validating " * i)

        try
            _ = CSV.read(annotations_dir * "/" * i, DataFrame)
        catch
            println("FAIL")
            continue
        end
        println("PASS")

    end
end
