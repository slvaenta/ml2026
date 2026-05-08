def grad_desc_1d(init, eta, eps):
    t = 1
    xt = init
    xT = xt - eta*(8*init+12)
    print("Итерация ", t)
    print("x0, x: ", xt, xT)
    print("Сравнение иксов: ", abs(xt - xT))
    while (abs(xt - xT) > eps):
        t+=1
        xt = xT
        xT = xt - eta*(8*xt+12)
        print("Итерация ", t)
        print("x0, x: ", xt, xT)
        print("Сравнение иксов: ", abs(xt - xT))
        
grad_desc_1d(0, 0.1, 0.1)

print("Пункт d, eta=0.01:")
grad_desc_1d(0, 0.01, 0.00001)
#114 итераций

print("Пункт d, eta=0.1:")
grad_desc_1d(0, 0.1, 0.00001)
#9 итераций

print("Пункт d, eta=0.2:")
grad_desc_1d(0, 0.2, 0.00001)
#26 итераций
"""
print("Пункт d, eta=0.3:")
grad_desc_1d(0, 0.3, 0.00001)
"""
    