using Luxor
using Colors

@png begin
cd(@__DIR__)    
Drawing(600, 600, "op_logo.svg")

p_size = 60

origin()
setcolor(1, 1, 1)
squircle(O, 300, 220, rt=0.5, action = :fill)

# Three points 
point1 = Point(-220, 10)
point2 = Point(10, 150)
point3 = Point(210, -100)

# First point
setcolor(Luxor.julia_green)
circle(point1, p_size, action = :fill)
setcolor(Luxor.julia_blue)
setline(12)
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
setline(10)
green_red = blend(point1, point2, Luxor.julia_green, Luxor.julia_red)
setblend(green_red)
line(point1, point2)
strokepath()

# Draw second line
red_purple = blend(point2, point3, Luxor.julia_red, Luxor.julia_purple)
setblend(red_purple)
line(point2, point3)
strokepath()

finish()
preview()
end
