#"""
#Name: 'M Mesh File (.m)...'
#Blender: 261
#Group: 'Export'
#Tooltip: 'Export M File'

# use
# Mesh: M_(MeshName)__Ignored
# Point: P_(Type)__Ignored

bl_info = {
   "name": "M Mesh File (.m)",
   "author": "Robert Crosby",
   "version": (1, 0, 0),
   "blender": (2, 6, 1),
   "api": 31847,
   "location": "File > Export > M Mesh File (.m)",
   "description": "Export M Mesh File (.m)",
   "warning": "",
   "wiki_url": "",
   "tracker_url": "",
   "category": "Import-Export"}

import bpy, struct, math, os, sys
from struct import*

#####################################
#Mesh exporter
#####################################

class vertex:
   def __init__(self, i, ver):
      self.index = i
      self.data = ver
      self.x = ver.co.x
      self.y = ver.co.y
      self.z = ver.co.z
   def writeVertex(self, out):
      out.write('Vertex %d  %f %f %f\n' % (self.index + 1, self.x , self.y, self.z))
   def duplicate(self, i):
      return vertex(i, self.data)


class face:
   def __init__(self, i, f):
      self.index = i
      self.data = f
      self.verts = ['', '', '']
      self.verts[0] = f.vertices[0]
      self.verts[1] = f.vertices[1]
      self.verts[2] = f.vertices[2]
   def writetofile(self, out):
      out.write('Face %d  %d %d %d\n' % (self.index + 1, self.verts[0] + 1, self.verts[1] + 1, self.verts[2] + 1))

def exportMesh(filePath):
   obj = bpy.context.active_object
   mesh = obj.data
   
   vertices = []
   for i, v in enumerate(mesh.vertices):
      vertices.append(vertex(i, v))
   
   faces = []
   for i, f in enumerate(mesh.faces):
      faces.append(face(i, f))
   
   out = open(filePath, "w")
   
   for v in vertices:
      v.writeVertex(out)
   
   for f in faces:
      f.writetofile(out)
   
   out.close()

#####################################
#class registration and interface
#####################################

from bpy.props import *
class ExportMap(bpy.types.Operator):
   '''Export M Mesh File (.m)'''
   bl_idname = "export.m"
   bl_label = 'Export M Mesh File'
   
   logenum = [("console", "Console", "log to console"),
              ("append", "Append", "append to log file"),
              ("overwrite", "Overwrite", "overwrite log file")]
   
   exportModes = []
   
   filepath = StringProperty(subtype='FILE_PATH', name="File Path", description="Filepath for exporting", maxlen=1024, default="")
   
   def execute(self, context):
      exportMesh(self.properties.filepath)
      return {'FINISHED'}
   
   def invoke(self, context, event):
      WindowManager = context.window_manager
      WindowManager.fileselect_add(self)
      return {"RUNNING_MODAL"}  

def menu_func(self, context):
   default_path = os.path.splitext(bpy.data.filepath)[0]
   self.layout.operator(ExportMap.bl_idname, text="M Mesh File (.m)", icon='BLENDER').filepath = default_path

def register():
   bpy.utils.register_module(__name__)
   bpy.types.INFO_MT_file_export.append(menu_func)

def unregister():
   bpy.utils.unregister_module(__name__)
   bpy.types.INFO_MT_file_export.remove(menu_func)

if __name__ == "__main__":
   register()
