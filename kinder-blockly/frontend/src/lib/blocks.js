import * as Blockly from 'blockly';

export function registerBlocks() {
  Blockly.Blocks['move_base_to_target'] = {
    init: function() {
      this.appendDummyInput()
        .appendField('Move base to')
        .appendField('x')
        .appendField(new Blockly.FieldNumber(0, -2, 2, 0.1), 'X')
        .appendField('y')
        .appendField(new Blockly.FieldNumber(0, -2, 2, 0.1), 'Y');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(260);
      this.setTooltip('Move the robot base to the given (x, y) position.');
    }
  };

  Blockly.Blocks['move_base_by'] = {
    init: function() {
      this.appendDummyInput()
        .appendField('Move base by')
        .appendField('x')
        .appendField(new Blockly.FieldNumber(0, -4, 4, 0.1), 'DX')
        .appendField('y')
        .appendField(new Blockly.FieldNumber(0, -4, 4, 0.1), 'DY');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#a88fe0');
      this.setTooltip('Move the robot base by (dx, dy) relative to its current position.');
    }
  };

  Blockly.Blocks['set_pen_color'] = {
    init: function() {
      const toHex = (r, g, b) => '#' + [r, g, b].map(v =>
        Math.max(0, Math.min(255, Math.round(Number(v)))).toString(16).padStart(2, '0')
      ).join('');

      function makeValidator(ch) {
        return function(newVal) {
          const src = this.getSourceBlock();
          const r = ch === 'R' ? Number(newVal) : Number(src.getFieldValue('R'));
          const g = ch === 'G' ? Number(newVal) : Number(src.getFieldValue('G'));
          const b = ch === 'B' ? Number(newVal) : Number(src.getFieldValue('B'));
          src.setColour(toHex(r, g, b));
          return newVal;
        };
      }

      const rField = new Blockly.FieldNumber(255, 0, 255, 1);
      const gField = new Blockly.FieldNumber(0,   0, 255, 1);
      const bField = new Blockly.FieldNumber(0,   0, 255, 1);
      rField.setValidator(makeValidator('R'));
      gField.setValidator(makeValidator('G'));
      bField.setValidator(makeValidator('B'));

      this.appendDummyInput()
        .appendField('Set pen color')
        .appendField('R').appendField(rField, 'R')
        .appendField('G').appendField(gField, 'G')
        .appendField('B').appendField(bField, 'B');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#ff0000'); // matches default R=255 G=0 B=0
      this.setTooltip('Set the drawing colour (RGB 0-255).');
    }
  };

  Blockly.Blocks['start'] = {
    init: function() {
      this.appendDummyInput()
        .appendField('Start');
      this.setNextStatement(true, null);
      this.setColour('#4ade80');
      this.setTooltip('Programs begin here. Only blocks connected below this will run.');
    }
  };

  Blockly.Blocks['pen_up'] = {
    init: function() {
      this.appendDummyInput()
        .appendField('Pen up');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip('Stop drawing while the robot moves.');
    }
  };

  Blockly.Blocks['pen_down'] = {
    init: function() {
      this.appendDummyInput()
        .appendField('Pen down');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip('Resume drawing while the robot moves.');
    }
  };
}

export const toolbox = {
  kind: 'categoryToolbox',
  contents: [
    { kind: 'category', name: 'Program', colour: '120', contents: [
      { kind: 'block', type: 'start' },
    ]},
    { kind: 'category', name: 'Movement', colour: '260', contents: [
      { kind: 'block', type: 'move_base_to_target' },
      { kind: 'block', type: 'move_base_by' },
    ]},
    { kind: 'category', name: 'Pen', colour: '310', contents: [
      { kind: 'block', type: 'set_pen_color' },
      { kind: 'block', type: 'pen_down' },
      { kind: 'block', type: 'pen_up' },
    ]},
  ]
};
