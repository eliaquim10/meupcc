print ((lambda x ,f  = lambda x: 2*x+1,
    g = lambda x: x**2 + 2*x,
    fg = lambda x: f(g(x)) :
    map(fg(x),range(1,11)
        ) (1)))
