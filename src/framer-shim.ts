// Shim of Framer SDK for local development and build compatibility
export const ControlType = {
  Array: 'array',
  Object: 'object',
  Number: 'number',
  String: 'string',
  Boolean: 'boolean',
  Image: 'image',
  ResponsiveImage: 'responsiveImage',
  File: 'file',
  Color: 'color',
  Enum: 'enum',
  Font: 'font',
  Transition: 'transition',
  ComponentInstance: 'componentInstance',
};

export function addPropertyControls(component: any, controls: any) {
  // Mock function - does nothing locally, but native in Framer editor
  return;
}
