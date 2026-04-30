import * as Blockly from 'blockly';

class FieldLabelUnderline extends Blockly.FieldLabel {
  initView() {
    super.initView();
    if (this.textElement_) this.textElement_.style.textDecoration = 'underline';
  }
}

class FieldLabelColored extends Blockly.FieldLabel {
  constructor(text, color) { super(text); this.color_ = color; }
  initView() {
    super.initView();
    if (this.textElement_) this.textElement_.style.fill = this.color_;
  }
}

class FieldColorSwatch extends Blockly.Field {
  constructor(r = 255, g = 0, b = 0) {
    super(`${r},${g},${b}`);
    this.SERIALIZABLE = true;
    this.r_ = r; this.g_ = g; this.b_ = b;
    this.size_ = new Blockly.utils.Size(24, 24);
  }
  initView() {
    super.initView();
    this.rect_ = Blockly.utils.dom.createSvgElement('rect',
      { width: '20', height: '20', rx: '3', x: '2', y: '2' },
      this.fieldGroup_,
    );
    this.rect_.setAttribute('data-cs', '1');
    this.rect_._csField = this;
    this.updateColor_();
  }
  doClassValidation_(val) { return val; }
  doValueUpdate_(val) {
    const parts = String(val).split(',').map(Number);
    if (parts.length === 3 && !parts.some(isNaN)) {
      [this.r_, this.g_, this.b_] = parts;
    }
    this.updateColor_();
  }
  render_() { super.render_(); this.updateColor_(); }
  updateColor_() {
    if (this.rect_) this.rect_.style.fill = `rgb(${this.r_},${this.g_},${this.b_})`;
  }
  setRGB(r, g, b) { this.setValue(`${r},${g},${b}`); }
}

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

  Blockly.Blocks['define_skill'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendDummyInput('HEADER')
        .appendField('Define skill')
        .appendField(new Blockly.FieldTextInput('my skill'), 'NAME')
        .appendField(new Blockly.FieldDropdown([
          ['0 params','0'],['1 param','1'],['2 params','2'],
          ['3 params','3'],['4 params','4'],['5 params','5'],
        ]), 'ARGS');
      this.setNextStatement(true, null);
      this.setColour('#1e3a8a');
      this.setTooltip('Define a reusable skill. Blocks connected below belong to this skill.');
      this.setOnChange(function(evt) {
        if (evt.type === Blockly.Events.BLOCK_CHANGE &&
            evt.blockId === this.id &&
            evt.element === 'field' && evt.name === 'ARGS') {
          this.updateShape_();
        }
      });
    },
    createParamInputs_: function(count) {
      const saved = [];
      for (let i = 0; this.getInput('PARAM' + i); i++) {
        saved[i] = {
          name: this.getFieldValue('PARAM_NAME_' + i),
          type: this.getFieldValue('PARAM_TYPE_' + i),
        };
        this.removeInput('PARAM' + i);
      }
      const types = [['integer','int'],['floating point','float'],['color','color']];
      for (let i = 0; i < count; i++) {
        this.appendDummyInput('PARAM' + i)
          .appendField('  • ')
          .appendField(new Blockly.FieldTextInput(saved[i]?.name ?? 'param' + (i + 1)), 'PARAM_NAME_' + i)
          .appendField(' : ')
          .appendField(new Blockly.FieldDropdown(types), 'PARAM_TYPE_' + i);
        if (saved[i]?.type) this.setFieldValue(saved[i].type, 'PARAM_TYPE_' + i);
      }
    },
    updateShape_: function() {
      const wasClosed = !!this.isCustomCollapsed_;
      if (wasClosed) this.customExpand_();
      this.createParamInputs_(parseInt(this.getFieldValue('ARGS')) || 0);
      if (wasClosed) this.customCollapse_();
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      const count = parseInt(this.getFieldValue('ARGS')) || 0;
      if (this.getInput('HEADER')) this.getInput('HEADER').setVisible(false);
      for (let i = 0; this.getInput('PARAM' + i); i++) this.getInput('PARAM' + i).setVisible(false);
      // Remove stale summary inputs
      if (this.getInput('SUM_HEADER')) this.removeInput('SUM_HEADER');
      for (let i = 0; this.getInput('SUM_' + i); i++) this.removeInput('SUM_' + i);
      if (this.getInput('SUM_FOOTER')) this.removeInput('SUM_FOOTER');
      // Build summary
      const name = this.getFieldValue('NAME') || 'my skill';
      if (count === 0) {
        this.appendDummyInput('SUM_HEADER')
          .appendField('def ')
          .appendField(new FieldLabelUnderline(name))
          .appendField('()');
      } else {
        this.appendDummyInput('SUM_HEADER')
          .appendField('def ')
          .appendField(new FieldLabelUnderline(name))
          .appendField('(');
        for (let i = 0; i < count; i++) {
          const pname = this.getFieldValue('PARAM_NAME_' + i) || ('param' + (i + 1));
          const ptype = this.getFieldValue('PARAM_TYPE_' + i) || 'int';
          const comma = i < count - 1 ? ',' : '';
          this.appendDummyInput('SUM_' + i).appendField('    ' + pname + ': ' + ptype + comma);
        }
        this.appendDummyInput('SUM_FOOTER').appendField(')');
      }
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM_HEADER')) this.removeInput('SUM_HEADER');
      for (let i = 0; this.getInput('SUM_' + i); i++) this.removeInput('SUM_' + i);
      if (this.getInput('SUM_FOOTER')) this.removeInput('SUM_FOOTER');
      if (this.getInput('HEADER')) this.getInput('HEADER').setVisible(true);
      for (let i = 0; this.getInput('PARAM' + i); i++) this.getInput('PARAM' + i).setVisible(true);
    },
    saveExtraState: function() {
      return { paramCount: parseInt(this.getFieldValue('ARGS')) || 0 };
    },
    loadExtraState: function(state) {
      this.createParamInputs_(state.paramCount || 0);
    },
  };

  Blockly.Blocks['use_skill'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.paramDefs_ = [];
      const getOptions = function() {
        const ws = this.getSourceBlock()?.workspace;
        const skills = ws?.getAllBlocks(false)
          .filter(b => b.type === 'define_skill')
          .map(b => b.getFieldValue('NAME') || 'unnamed')
          .filter((n, i, a) => a.indexOf(n) === i);
        return skills?.length ? skills.map(n => [n, n]) : [['(no skills)', '__NONE__']];
      };
      this.appendDummyInput('HEADER')
        .appendField('Use skill')
        .appendField(new Blockly.FieldDropdown(getOptions), 'SKILL');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#14b8a6');
      this.setTooltip('Execute a defined skill.');
      this.setOnChange(function(evt) {
        if (evt.type === Blockly.Events.BLOCK_CHANGE &&
            evt.blockId === this.id &&
            evt.element === 'field') {
          if (evt.name === 'SKILL') {
            this.updateParamInputs_();
          } else if (evt.name?.startsWith('ARG_COLOR_')) {
            const idx = parseInt(evt.name.slice('ARG_COLOR_'.length));
            if (!isNaN(idx) && this.staleSaved_?.has(idx)) {
              this.staleSaved_.delete(idx);
              this.setFieldValue('', 'ARG_STALE_' + idx);
            }
          }
        }
      });
    },
    createParamInputsFromDefs_: function(defs) {
      for (let i = 0; this.getInput('ARG_' + i); i++) this.removeInput('ARG_' + i);
      for (let i = 0; i < defs.length; i++) {
        const { name, type } = defs[i];
        const row = this.appendDummyInput('ARG_' + i).appendField('  ' + name + ': ');
        if (type === 'int' || type === 'float') row.appendField(new Blockly.FieldTextInput('NULL'), 'ARG_VAL_' + i);
        if (type === 'color') {
          row.appendField(new FieldColorSwatch(), 'ARG_COLOR_' + i);
          row.appendField(new Blockly.FieldLabel(''), 'ARG_STALE_' + i);
        }
      }
    },
    updateParamInputs_: function() {
      const wasClosed = !!this.isCustomCollapsed_;
      if (wasClosed) this.customExpand_();

      // Snapshot current values before rebuilding; prefer stored imprecise value over zeroed field
      const saved = [];
      for (let i = 0; this.getInput('ARG_' + i); i++) {
        const nf = this.getField('ARG_VAL_' + i);
        const cf = this.getField('ARG_COLOR_' + i);
        if (nf) {
          const imprecise = this.impreciseSaved_?.[i];
          saved[i] = { kind: 'num', val: imprecise !== undefined ? imprecise : nf.getValue() };
        } else if (cf) {
          saved[i] = { kind: 'color', r: cf.r_, g: cf.g_, b: cf.b_ };
        }
      }

      const skillName = this.getFieldValue('SKILL');
      const skillBlock = this.workspace?.getAllBlocks(false).find(
        b => b.type === 'define_skill' && b.getFieldValue('NAME') === skillName
      );
      const count = parseInt(skillBlock?.getFieldValue('ARGS')) || 0;
      this.paramDefs_ = [];
      for (let i = 0; i < count; i++) {
        this.paramDefs_.push({
          name: skillBlock.getFieldValue('PARAM_NAME_' + i) || ('param' + (i + 1)),
          type: skillBlock.getFieldValue('PARAM_TYPE_' + i) || 'int',
        });
      }
      this.createParamInputsFromDefs_(this.paramDefs_);

      // Restore values where the new type is compatible; flag precision loss or kind mismatch
      this.impreciseSaved_ = {};
      this.staleSaved_ = new Set();
      for (let i = 0; i < this.paramDefs_.length; i++) {
        const s = saved[i];
        if (!s) continue;
        const { type } = this.paramDefs_[i];
        if (s.kind === 'num' && (type === 'int' || type === 'float')) {
          const isNull = String(s.val).toUpperCase() === 'NULL';
          const num = Number(s.val);
          if (!isNull && type === 'int' && num !== Math.round(num)) {
            this.impreciseSaved_[i] = s.val;
          } else {
            this.getField('ARG_VAL_' + i)?.setValue(s.val);
          }
        } else if (s.kind === 'color' && type === 'color') {
          this.getField('ARG_COLOR_' + i)?.setRGB(s.r, s.g, s.b);
        } else {
          this.staleSaved_.add(i);
          if (this.paramDefs_[i]?.type === 'color') {
            this.setFieldValue(' NULL', 'ARG_STALE_' + i);
          }
        }
      }

      if (wasClosed) this.customCollapse_();
    },
    saveExtraState: function() {
      return { paramDefs: this.paramDefs_ || [] };
    },
    loadExtraState: function(state) {
      this.paramDefs_ = state.paramDefs || [];
      this.createParamInputsFromDefs_(this.paramDefs_);
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      if (this.getInput('HEADER')) this.getInput('HEADER').setVisible(false);
      for (let i = 0; this.getInput('ARG_' + i); i++) this.getInput('ARG_' + i).setVisible(false);
      if (this.getInput('SUM_HEADER')) this.removeInput('SUM_HEADER');
      for (let i = 0; this.getInput('SUM_' + i); i++) this.removeInput('SUM_' + i);
      if (this.getInput('SUM_FOOTER')) this.removeInput('SUM_FOOTER');
      const name = this.getFieldValue('SKILL') || 'skill';
      const defs = this.paramDefs_ || [];
      const skillBlock = this.workspace?.getAllBlocks(false).find(
        b => b.type === 'define_skill' && b.getFieldValue('NAME') === name
      );
      const skillCount = parseInt(skillBlock?.getFieldValue('ARGS')) || 0;
      const nameInvalid = !skillBlock || defs.length !== skillCount;
      const nameField = nameInvalid
        ? new FieldLabelColored(name, '#f87171')
        : new FieldLabelUnderline(name);
      if (defs.length === 0) {
        this.appendDummyInput('SUM_HEADER')
          .appendField(nameField).appendField('()');
      } else {
        this.appendDummyInput('SUM_HEADER')
          .appendField(nameField).appendField('(');
        for (let i = 0; i < defs.length; i++) {
          const { name: pname, type } = defs[i];
          let val;
          if (type === 'color' && this.staleSaved_?.has(i)) {
            val = 'NULL';
          } else if (type === 'color') {
            const f = this.getField('ARG_COLOR_' + i);
            val = f ? `rgb(${f.r_},${f.g_},${f.b_})` : '?';
          } else {
            const imprecise = this.impreciseSaved_?.[i];
            val = imprecise !== undefined ? imprecise : (this.getFieldValue('ARG_VAL_' + i) ?? '?');
          }
          const isNull = String(val).toUpperCase() === 'NULL';
          const comma = i < defs.length - 1 ? ',' : '';
          const text = '    ' + pname + ' = ' + val + comma;
          const expName = skillBlock?.getFieldValue('PARAM_NAME_' + i) || ('param' + (i + 1));
          const expType = skillBlock?.getFieldValue('PARAM_TYPE_' + i) || 'int';
          const bad = !skillBlock || i >= skillCount || pname !== expName || type !== expType
            || this.impreciseSaved_?.[i] !== undefined || this.staleSaved_?.has(i) || isNull;
          this.appendDummyInput('SUM_' + i)
            .appendField(bad ? new FieldLabelColored(text, '#f87171') : text);
        }
        this.appendDummyInput('SUM_FOOTER').appendField(')');
      }
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM_HEADER')) this.removeInput('SUM_HEADER');
      for (let i = 0; this.getInput('SUM_' + i); i++) this.removeInput('SUM_' + i);
      if (this.getInput('SUM_FOOTER')) this.removeInput('SUM_FOOTER');
      if (this.getInput('HEADER')) this.getInput('HEADER').setVisible(true);
      for (let i = 0; this.getInput('ARG_' + i); i++) this.getInput('ARG_' + i).setVisible(true);
    },
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
    { kind: 'category', name: 'Abstraction', colour: '#1e3a8a', contents: [
      { kind: 'block', type: 'define_skill' },
      { kind: 'block', type: 'use_skill' },
    ]},
  ]
};
