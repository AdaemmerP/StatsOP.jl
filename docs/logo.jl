using Luxor
using Colors

@png begin
Drawing(600, 600, "op_logo.svg")

p_size = 60

origin()
setcolor(0.2, 0.2, 0.3)
squircle(O, 300, 220, rt=0.5, action = :fill)

# Write package name
#fontsize(68)
#setcolor(Luxor.julia_purple)  
#text("OrdinalPatterns.jl", Point(0, 220), halign=:center, valign=:bottom)

# Three points 
point1 = Point(-210, -135)
point2 = Point(-20, 150)
point3 = Point(230, 0)

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

# Draw lines to connect the points
setcolor(Luxor.julia_blue)
setline(8)
line(point1, point2, action = :stroke)
line(point2, point3, action = :stroke)

finish()
preview()
end

