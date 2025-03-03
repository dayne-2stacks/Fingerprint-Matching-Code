import passiogo

system = passiogo.getSystemFromID(2343)

routes = system.getRoutes()

for i in range(len(routes)):
    print(routes[i].__dict__)
    print()


# stops = system.getStops()

# for i in range(len(stops)):
#     print(stops[i].__dict__)