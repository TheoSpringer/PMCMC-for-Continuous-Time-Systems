# julia.jl  — minimal plotting smoke test

using Plots
gr()  # use GR backend

x = range(0, 10; length=400)
y = @. sin(x) + 0.2cos(3x)

p = plot(x, y;
    xlabel="x",
    ylabel="y",
    title="Plots.jl smoke test",
    lw=3,
    label="sin(x) + 0.2cos(3x)"
)

display(p)
try
    gui(p)
catch
end
println("Plotting test completed successfully.")