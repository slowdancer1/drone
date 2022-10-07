import sys

if len(sys.argv) > 1:
    name = sys.argv[1]
    obj_file = open("{}.obj".format(name))
    mtl_file = open("{}.mtl".format(name))

    h_file = open("{}.h".format(name), 'w')
    h_file.write("#ifndef {}_3D_H\n".format(name.upper()))
    h_file.write("#define {}_3D_H\n\n".format(name.upper()))
    obj_lines = [line.strip() for line in obj_file.readlines()]
    mtl_lines = [line.strip() for line in mtl_file.readlines()]
    h_file.write('#define {}_3D_OBJ "{}"\n\n'.format(name.upper(), '\\n'.join(obj_lines)))
    h_file.write('#define {}_3D_MTL "{}"\n\n'.format(name.upper(), '\\n'.join(mtl_lines)))
    h_file.write("#endif\n")
    h_file.close()

    obj_file.close()
    mtl_file.close()
