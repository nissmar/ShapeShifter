# Process your own data


## Files

- Place the watertight `NAME.obj` file in `data/GT_WT`
- Place the textured `NAME.glb` file in `data/GT_GLB`
- Add the material `NAME.mtl` file in `data/materials` (or copy and rename `default.mtl`)

## From mesh to VDB tensor

Full encoding:
```shell
 cd src/shape_encoding/
 python shape_encoding.py -name NAME 
 ```

Encoding without color:
```shell
 cd src/shape_encoding/
 python shape_encoding.py -name NAME -no_color True
 ```


