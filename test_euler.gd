extends SceneTree

func _init():
    var basis = Basis()
    basis = basis.rotated(Vector3(1, 0, 0), 1.0)
    var v = basis * Vector3(0, -1, 0)
    print("Vector Y(-1) rotated by +1 rad around X: ", v)
    
    var euler = basis.get_euler()
    print("Euler: ", euler)
    quit()
