from rollstats import Container

if __name__ == "__main__":
    container = Container(window_size=5)
    container.subscribe_std()
    for i in range(10):
        container.push(i)
        print(f"n: {container.n.value}, std: {container.std.value:.2f}")
