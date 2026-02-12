if __name__ == "__main__":
    import jello.fem1d
    import jello.j1
    import jello.j1_otimes_j1
    import jello.measure
    import jello.phi2d
    import jello.phi1d
    # import jello.j1_otimes_j1_from_function
    import jello.fem2d
    import jello.body

    jello.j1.test()
    jello.j1_otimes_j1.test()
    jello.phi2d.test()
    jello.phi1d.test()
    jello.fem1d.test()
    # jello.j1_otimes_j1_from_function.test()
    jello.measure.test()
    jello.fem2d.test()
    jello.body.test()

    print('jello.test -- all tests passed')
