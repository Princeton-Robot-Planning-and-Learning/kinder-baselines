/* Custom Blockly block definitions for KinDER skills. */

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
        this.setColour(210);
        this.setTooltip('Move the robot base to the given (x, y) position.');
    }
};

Blockly.Blocks['set_pen_color'] = {
    init: function() {
        this.appendDummyInput()
            .appendField('Set pen color')
            .appendField('R')
            .appendField(new Blockly.FieldNumber(255, 0, 255, 1), 'R')
            .appendField('G')
            .appendField(new Blockly.FieldNumber(0, 0, 255, 1), 'G')
            .appendField('B')
            .appendField(new Blockly.FieldNumber(0, 0, 255, 1), 'B');
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setColour(20);
        this.setTooltip('Set the drawing colour (RGB 0-255). Puts the pen down.');
    }
};

Blockly.Blocks['pen_up'] = {
    init: function() {
        this.appendDummyInput()
            .appendField('Pen up');
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setColour(20);
        this.setTooltip('Stop drawing while the robot moves.');
    }
};

Blockly.Blocks['pen_down'] = {
    init: function() {
        this.appendDummyInput()
            .appendField('Pen down');
        this.setPreviousStatement(true, null);
        this.setNextStatement(true, null);
        this.setColour(20);
        this.setTooltip('Resume drawing while the robot moves.');
    }
};
