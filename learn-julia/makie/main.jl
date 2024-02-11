using GLMakie
using ArgParse

function parse_cli()
end

function julia(z::ComplexF32, c::ComplexF32)::Float32
    for i in 1:70.0; abs(z) > 2 && return i; z = z^2 + c; end; 0
end

function get_data(xs::Vector{Float32}, ys::Vector{Float32}, c::ComplexF32)::Array{Float32}
    res = Array{Float32}(undef, length(xs), length(ys))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            res[i,j] = julia(x+y*im, c)
        end
    end
    return res
end

function get_fig(c::ComplexF32)
    n_points = 500
    xs = Float32.(collect(LinRange(-1.5, 1.5, n_points)))
    ys = Float32.(collect(LinRange(-1.5, 1.5, n_points)))
    println("Getting Data")
    zs = get_data(xs, ys, c)
    println("Plotting Data")
    return heatmap(xs, ys, zs, colormap = Reverse(:deep); figure = (; resolution = (900, 900)))
end


function main()
    s = ArgParseSettings()
    @add_arg_table s begin
        "cx"
            help = "a positional argument"
            arg_type = Float64
            default = -0.1
        "cy"
            help = "a positional argument"
            arg_type = Float64
            default = 0.65
    end
    args = parse_args(ARGS, s)

    c = Float32(args["cx"]) + Float32(args["cy"]) * im
    fig, ax, p = get_fig(c);

    # %% Zooming and recompute functionality
    on(events(ax.scene).mousebutton, priority = 2) do event
        if event.button == Mouse.left
            mp = mouseposition(ax.scene)

            if event.action == Mouse.press
                global points = Array{Float32}(undef, 2, 2)
                points[1,:] = mp
            elseif event.action == Mouse.release
                points[2,:] = mp
                x_min, x_max = minimum(points[:, 1]), maximum(points[:,1])
                y_min, y_max = minimum(points[:, 2]), maximum(points[:,2])

                n_points = 500
                xs = collect(range(x_min,x_max,n_points))
                ys = collect(range(y_min,y_max,n_points))
                zs = get_data(xs, ys, c)
                heatmap!(ax, xs, ys, zs, colormap = Reverse(:deep))
            end

            return Consume(false)
        end
    end

    return fig
end

fig = main()
display(fig)
print("Press any key to contnue. ")
readline()


