module Tmp

using GLMakie

function foo1()
    x = 1:0.1:10
    fig = lines(x, x.^2; label = "Parabola",
        axis = (; xlabel = "x", ylabel = "y", title ="Title"),
        figure = (; resolution = (800,600), fontsize = 22))
    axislegend(; position = :lt)
    save("./assets/parabola.png", fig)
    return fig
end

function foo2()
    points = Observable(Point2f[])

    scene = Scene(camera = campixel!)
    linesegments!(scene, points, color = :black)
    scatter!(scene, points, color = :gray)

    on(events(scene).mousebutton) do event
        if event.button == Mouse.left
            if event.action == Mouse.press || event.action == Mouse.release
                mp = events(scene).mouseposition[]
                push!(points[], mp)
                notify(points)
            end
        end
    end

    return scene
end

function julia(x, y)
    c = -0.1 + 0.65 * im
    z = x + y * im
    for i in 1:50.0; abs(z) > 2 && return i; z = z^2 + c; end; 0
end

function get_data(xs, ys)
    res = Array{Float64}(undef, length(xs), length(ys))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            res[i,j] = julia(x, y)
        end
    end
    return res
end

function foo3()
    xs = collect(-1.5:0.001:1.5)
    ys = collect(-1.5:0.001:1.5)
    println("Getting Data")
    zs = get_data(xs, ys)
    println("Plotting Data")
    return heatmap(xs, ys, zs, colormap = Reverse(:deep))
end

function main()
    return foo3()
end

end
