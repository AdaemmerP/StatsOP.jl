using Luxor
using Colors

@png begin
Drawing(600, 600, "op_logo.svg")

p_size = 50

origin()
setcolor(0.2, 0.2, 0.3)
squircle(O, 300, 300, rt=0.5, action = :fill)

# Write package name
setcolor(Luxor.julia_blue)
fontsize(65)
text("OrdinalPatterns.jl", Point(0, -150), halign=:center, valign=:bottom)

# Three points 
point1 = Point(-220, -50)
point2 = Point(0, 200)
point3 = Point(240, 30)

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
setline(5)
line(point1, point2, action = :stroke)
line(point2, point3, action = :stroke)
fontsize(20)
# Set font size for the title
fontsize(30)
# Draw the title at a specified position
#text("OrdinalPatterns.jl", Point(200, 0), halign=:center, valign=:bottom)
finish()
#Luxor.saveimage("my-drawing.svg")
preview()
end

