using Luxor
using Colors

@png begin
    cd(@__DIR__)
    Drawing(600, 600, "op_logo.svg")

    p_size = 60

    origin()
    setcolor(.95,.95, .95)
    squircle(O, 300, 220, rt=0.5, action = :fill)

    # Three points 
    point1 = Point(-233, 10)
    point2 = Point(10, 150)
    point3 = Point(220, -120)

    # First point
    setline(15) # Line width for :stroke
    setcolor(Luxor.julia_red)
    circle(point1, p_size, action=:fill)
    setcolor(Luxor.julia_blue)
    circle(point1, p_size, action=:stroke)

    # Second point
    setcolor(Luxor.julia_green)
    circle(point2, p_size, action=:fill)
    setcolor(Luxor.julia_blue)
    circle(point2, p_size, action=:stroke)

    # Third point
    setcolor(Luxor.julia_purple)
    circle(point3, p_size, action=:fill)
    setcolor(Luxor.julia_blue)
    circle(point3, p_size, action=:stroke)

    # # Blend to fill
    # green_red = blend(Point(0, 0), 5, Point(0, 0), 250, Luxor.julia_green, Luxor.julia_red)
    # red_green = blend(Point(0, 0), 5, Point(0, 0), 250, Luxor.julia_red, Luxor.julia_green)
    # green_purple = blend(Point(0, 0), 5, Point(0, 0), 250, Luxor.julia_green, Luxor.julia_purple)
    # # First point
    # setblend(green_red)
    # circle(point1, p_size, action=:fill)
    # # Second point
    # setblend(red_green)    
    # circle(point2, p_size, action=:fill)
    # # Third point
    # setblend(green_purple)
    # circle(point3, p_size, action=:fill)

    # Draw first lines
    setline(20)
    green_red = blend(point1, point2, Luxor.julia_red, Luxor.julia_green)
    setblend(green_red)
    line(point1, point2)
    strokepath()

    # Draw second line
    red_purple = blend(point2, point3, Luxor.julia_green, Luxor.julia_purple)
    setblend(red_purple)
    line(point2, point3)
    strokepath()

    finish()
    preview()
end
