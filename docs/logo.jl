using Luxor
using Colors

@png begin
Drawing(600, 600, "op_logo.svg")

p_size = 60

origin()
setcolor(1, 1, 1)
squircle(O, 300, 220, rt=0.5, action = :fill)

# Three points 
point1 = Point(-190, -110)
point2 = Point(-10, 130)
point3 = Point(210, -20)

# First point
setcolor(Luxor.julia_green)
circle(point1, p_size, action = :fill)
setcolor(Luxor.julia_blue)
circle(point1, p_size, action = :stroke)

# Second point
setcolor(Luxor.julia_red)
circle(point2, p_size, action = :fill)
setcolor(Luxor.julia_blue)
circle(point2, p_size, action = :stroke)

# Third point
setcolor(Luxor.julia_purple)    
circle(point3, p_size, action = :fill)
setcolor(Luxor.julia_blue)
circle(point3, p_size, action = :stroke)

# Draw first lines
setline(8)
green_red = blend(point1, point2, Luxor.julia_green, Luxor.julia_red)
setblend(green_red)
line(point1, point2)
strokepath()

# Draw second line
red_purple = blend(point2, point3, Luxor.julia_red, Luxor.julia_purple)
setblend(red_purple)
line(point2, point3)
strokepath()


# setcolor(Luxor.julia_blue)
# setline(8)
# line(point1, point2, action = :stroke)
# line(point2, point3, action = :stroke)

finish()
preview()
end

logo = begin
    Drawing(200, 200, "logo.svg")
    origin()
    scale(3.4)

    fontface("Arial Bold")
    fontsize(50)

    # Define center points and corners of triangle
    center = Point(0, 4)
    corners = ngon(center, 16, 3, -0.75π / 2; vertices=true)

    # Draw three blended partials
    for (i, c) in enumerate(corners)
        b = blend(c, 0, center, 50, colors[i], "black")
        setblend(b)
        text("∂", c; valign=:middle, halign=:center, angle=0.68π + 2π / 3 * i)
    end

    finish()
    preview()
end
