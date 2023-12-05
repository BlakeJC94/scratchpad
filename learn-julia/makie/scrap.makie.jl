using GLMakie

function julia(x, y)
    c = -0.1 + 0.65 * im
    z = x + y * im
    for i in 1:70.0; abs(z) > 2 && return i; z = z^2 + c; end; 0
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

function get_fig()
    n_points = 500
    xs = collect(LinRange(-1.5, 1.5, n_points))
    ys = collect(LinRange(-1.5, 1.5, n_points))
    println("Getting Data")
    zs = get_data(xs, ys)
    println("Plotting Data")
    return heatmap(xs, ys, zs, colormap = Reverse(:deep); figure = (; resolution = (900, 900)))
end

# %% Zooming and recompute functionality
fig, ax, p = get_fig();

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
            zs = get_data(xs, ys)
            heatmap!(ax, xs, ys, zs, colormap = Reverse(:deep))
        end

        return Consume(false)
    end
end

fig


# %% Sliders to tweak the shape

f = Figure(resolution=(900,900))
Axis(f[1, 1])

rs_h = Slider(f[2, 1], range = LinRange(-2, 2, 1000), startvalue = 0.1)
rs_v = Slider(f[1, 2], range = LinRange(-2, 2, 1000), startvalue = 0.65, horizontal = false)

labeltext1 = lift(rs_h.value) do int
    string(round.(int, digits = 2))
end
Label(f[3, 1], labeltext1, tellwidth = false)

labeltext2 = lift(rs_v.value) do int
    string(round.(int, digits = 2))
end
Label(f[1, 3], labeltext2, tellheight = false, rotation = pi/2)

n_points = 500
xs = collect(LinRange(-1.5, 1.5, n_points))
ys = collect(LinRange(-1.5, 1.5, n_points))

function julia(x, y, c)
    z = x + y * im
    for i in 1:70.0; abs(z) > 2 && return i; z = z^2 + c; end; 0
end

function get_data(xs, ys, c=-0.1 + 0.65*im)
    res = Array{Float64}(undef, length(xs), length(ys))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            res[i,j] = julia(x, y, c)
        end
    end
    return res
end

zs = lift(rs_h.value, rs_v.value) do λ, μ
    c = λ + μ * im
    get_data(xs, ys, c)
end

heatmap!(xs, ys, zs);
f

# %% Altogether??

f = Figure(resolution=(900,900))
ax = Axis(f[1, 1])

rs_h = Slider(f[2, 1], range = LinRange(-2, 2, 1000), startvalue = 0.1)
rs_v = Slider(f[1, 2], range = LinRange(-2, 2, 1000), startvalue = 0.65, horizontal = false)

labeltext1 = lift(rs_h.value) do int
    string(round.(int, digits = 2))
end
Label(f[3, 1], labeltext1, tellwidth = false)

labeltext2 = lift(rs_v.value) do int
    string(round.(int, digits = 2))
end
Label(f[1, 3], labeltext2, tellheight = false, rotation = pi/2)

n_points = 500
xs = collect(LinRange(-1.5, 1.5, n_points))
ys = collect(LinRange(-1.5, 1.5, n_points))

function julia(x, y, c)
    z = x + y * im
    for i in 1:70.0; abs(z) > 2 && return i; z = z^2 + c; end; 0
end

function get_data(xs, ys, c=-0.1 + 0.65*im)
    res = Array{Float64}(undef, length(xs), length(ys))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            res[i,j] = julia(x, y, c)
        end
    end
    return res
end

zs = lift(rs_h.value, rs_v.value) do a, b
    c = a + b * im
    get_data(xs, ys, c)
end

heatmap!(xs, ys, zs, colormap = Reverse(:deep));

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
            zs = lift(rs_h.value, rs_v.value) do a, b
                c = a + b * im
                get_data(xs, ys, c)
            end
            heatmap!(ax, xs, ys, zs, colormap = Reverse(:deep))
        end

        return Consume(false)
    end
end
f
