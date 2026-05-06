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
    if (this.textElement_) this.textElement_.style.setProperty('fill', this.color_, 'important');
  }
  applyColour() {
    super.applyColour();
    if (this.textElement_) this.textElement_.style.setProperty('fill', this.color_, 'important');
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

class FieldDropdownDark extends Blockly.FieldDropdown {
  // Accept any non-null value so dynamic dropdowns survive workspace load
  // order (the saved value may not be in options yet when the field is restored).
  doClassValidation_(newVal) { return newVal ?? null; }
  initView() {
    super.initView();
    this.forceTextDark_();
  }
  applyColour() {
    super.applyColour();
    this.forceTextDark_();
    if (this.arrow) this.arrow.style.setProperty('fill', '#1a1a1a', 'important');
  }
  render_() {
    super.render_();
    this.forceTextDark_();
  }
  showEditor_(opt_e) {
    super.showEditor_(opt_e);
    requestAnimationFrame(() => {
      document.querySelector('.blocklyDropDownDiv')?.classList.add('kinder-param-dropdown');
    });
  }
  forceTextDark_() {
    this.fieldGroup_?.querySelectorAll('text, tspan').forEach(el => {
      el.style.setProperty('fill', '#1a1a1a', 'important');
    });
    if (this.textElement_) this.textElement_.style.setProperty('fill', '#1a1a1a', 'important');
  }
}

// Like FieldDropdownDark but without forced dark text — used on dark-coloured blocks.
class FieldDropdownPermissive extends Blockly.FieldDropdown {
  doClassValidation_(newVal) { return newVal ?? null; }
}

// Walk up the parent chain to find the topmost block.
function getChainHead(block) {
  let b = block;
  while (b) {
    const parent = b.getParent();
    if (!parent) return b;
    b = parent;
  }
  return null;
}

export function registerBlocks() {
  // Inline number/param block used as a shadow inside movement value inputs.
  // Accepts a plain number OR a parameter name (resolved at execution time).
  Blockly.Blocks['kinder_num'] = {
    init: function() {
      this.appendDummyInput()
        .appendField(new Blockly.FieldTextInput('0'), 'NUM');
      this.setOutput(true, 'Number');
      this.setColour(260);
      this.setTooltip('A number value or parameter name (e.g. "x", "count").');
    }
  };

  // Parameter reference block — yellow, dropdown of params from enclosing define_skill.
  Blockly.Blocks['param_ref'] = {
    init: function() {
      const getOptions = function() {
        const block = this.getSourceBlock();
        if (!block) return [['(no params)', '__NONE__']];
        const head = getChainHead(block);
        if (head?.type !== 'define_skill') return [['(no params)', '__NONE__']];

        // Filter params by the expected type of the parent input.
        let expectedKind = null;
        const parent = block.getParent();
        if (parent) {
          if (parent.type === 'set_pen_color') expectedKind = 'color';
          else if (parent.type === 'move_base_to_target' || parent.type === 'move_base_by'
                || parent.type === 'repeat' || parent.type === 'condition') expectedKind = 'num';
        }

        const count = parseInt(head.getFieldValue('ARGS')) || 0;
        const opts = [];
        for (let i = 0; i < count; i++) {
          const name = head.getFieldValue('PARAM_NAME_' + i) || ('param' + (i + 1));
          const type = head.getFieldValue('PARAM_TYPE_' + i) || 'int';
          const isNum = type === 'int' || type === 'float';
          const isColor = type === 'color';
          if (expectedKind === 'num' && !isNum) continue;
          if (expectedKind === 'color' && !isColor) continue;
          opts.push([name, name]);
        }
        return opts.length ? opts : [['(no params)', '__NONE__']];
      };
      this.appendDummyInput()
        .appendField(new FieldDropdownDark(getOptions), 'PARAM');
      this.setOutput(true, 'Param');
      this.setStyle('param_style');
      this.setTooltip('Reference a parameter from the enclosing skill.');
    }
  };

  function collapseValueInput(block, inputName) {
    const connected = block.getInput(inputName)?.connection?.targetBlock();
    if (!connected) return { text: '?', isParam: false };
    if (connected.type === 'param_ref') return { text: connected.getFieldValue('PARAM') || '?', isParam: true };
    const raw = String(connected.getFieldValue('NUM') ?? '?').trim();
    // Treat non-numeric text as a parameter name reference
    const isParam = raw !== '?' && raw !== '' && isNaN(Number(raw));
    return { text: raw, isParam };
  }

  Blockly.Blocks['move_base_to_target'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendValueInput('INPUT_X').setCheck(['Number', 'Param']).appendField('Move base to x');
      this.appendValueInput('INPUT_Y').setCheck(['Number', 'Param']).appendField('y');
      this.setInputsInline(true);
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(260);
      this.setTooltip('Move the robot base to the given (x, y) position.');
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.getInput('INPUT_X')?.setVisible(false);
      this.getInput('INPUT_Y')?.setVisible(false);
      if (this.getInput('SUM')) this.removeInput('SUM');
      const x = collapseValueInput(this, 'INPUT_X');
      const y = collapseValueInput(this, 'INPUT_Y');
      const row = this.appendDummyInput('SUM').appendField('move base to (');
      row.appendField(x.isParam ? new FieldLabelUnderline(x.text) : x.text);
      row.appendField(', ');
      row.appendField(y.isParam ? new FieldLabelUnderline(y.text) : y.text);
      row.appendField(')');
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM')) this.removeInput('SUM');
      this.getInput('INPUT_X')?.setVisible(true);
      this.getInput('INPUT_Y')?.setVisible(true);
    },
  };

  Blockly.Blocks['move_base_by'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendValueInput('INPUT_DX').setCheck(['Number', 'Param']).appendField('Move base by x');
      this.appendValueInput('INPUT_DY').setCheck(['Number', 'Param']).appendField('y');
      this.setInputsInline(true);
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#a88fe0');
      this.setTooltip('Move the robot base by (dx, dy) relative to its current position.');
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.getInput('INPUT_DX')?.setVisible(false);
      this.getInput('INPUT_DY')?.setVisible(false);
      if (this.getInput('SUM')) this.removeInput('SUM');
      const dx = collapseValueInput(this, 'INPUT_DX');
      const dy = collapseValueInput(this, 'INPUT_DY');
      const row = this.appendDummyInput('SUM').appendField('move base by (');
      row.appendField(dx.isParam ? new FieldLabelUnderline(dx.text) : dx.text);
      row.appendField(', ');
      row.appendField(dy.isParam ? new FieldLabelUnderline(dy.text) : dy.text);
      row.appendField(')');
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM')) this.removeInput('SUM');
      this.getInput('INPUT_DX')?.setVisible(true);
      this.getInput('INPUT_DY')?.setVisible(true);
    },
  };

  Blockly.Blocks['set_pen_color'] = {
    init: function() {
      const toHex = (r, g, b) => '#' + [r, g, b].map(v =>
        Math.max(0, Math.min(255, Math.round(Number(v)))).toString(16).padStart(2, '0')
      ).join('');

      function makeValidator(ch) {
        return function(newVal) {
          const src = this.getSourceBlock();
          if (!src) return newVal;
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

      this.appendValueInput('COLOR_PARAM').setCheck(['Param']).appendField('Set pen color');
      this.appendDummyInput('RGB_ROW')
        .appendField('R').appendField(rField, 'R')
        .appendField('G').appendField(gField, 'G')
        .appendField('B').appendField(bField, 'B');
      this.setInputsInline(true);
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#ff0000');
      this.setTooltip('Set the drawing colour (RGB 0-255 or a color parameter).');

      this.setOnChange(function(evt) {
        if ([Blockly.Events.BLOCK_MOVE, Blockly.Events.BLOCK_CHANGE,
             Blockly.Events.BLOCK_CREATE, Blockly.Events.BLOCK_DELETE].includes(evt.type)) {
          if (!this.isCustomCollapsed_) {
            const hasParam = !!this.getInput('COLOR_PARAM')?.connection?.targetBlock();
            this.getInput('RGB_ROW')?.setVisible(!hasParam);
          }
        }
      });
      this.isCustomCollapsed_ = false;
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.getInput('COLOR_PARAM')?.setVisible(false);
      this.getInput('RGB_ROW')?.setVisible(false);
      if (this.getInput('SUM')) this.removeInput('SUM');
      const colorBlock = this.getInput('COLOR_PARAM')?.connection?.targetBlock();
      if (colorBlock) {
        const paramName = colorBlock.getFieldValue('PARAM') || '?';
        this.appendDummyInput('SUM')
          .appendField('set_pen_color (')
          .appendField(new FieldLabelUnderline(paramName))
          .appendField(')');
      } else {
        const r = Math.round(Number(this.getFieldValue('R')));
        const g = Math.round(Number(this.getFieldValue('G')));
        const b = Math.round(Number(this.getFieldValue('B')));
        this.appendDummyInput('SUM').appendField(`set_pen_color (${r}, ${g}, ${b})`);
      }
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM')) this.removeInput('SUM');
      this.getInput('COLOR_PARAM')?.setVisible(true);
      const hasParam = !!this.getInput('COLOR_PARAM')?.connection?.targetBlock();
      this.getInput('RGB_ROW')?.setVisible(!hasParam);
    },
  };

  Blockly.Blocks['start'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendDummyInput('MAIN').appendField(new Blockly.FieldLabel('Start'), 'LABEL');
      this.appendStatementInput('BODY');
      this.setColour('#4ade80');
      this.setTooltip('Programs begin here. Only blocks connected below this will run.');
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.setFieldValue('if __name__ == "__main__":', 'LABEL');
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      this.setFieldValue('Start', 'LABEL');
    },
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
      this.appendStatementInput('BODY');
      this.setColour('#1e3a8a');
      this.setTooltip('Define a reusable skill. Blocks inside belong to this skill.');
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
      // Keep BODY at the end (params were appended after it).
      this.moveInputBefore('BODY', null);
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
      // Keep BODY after the summary rows.
      this.moveInputBefore('BODY', null);
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
        .appendField(new FieldDropdownPermissive(getOptions), 'SKILL');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#0f766e');
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
      // Skill not found (e.g. loading before define_skill exists) — keep existing state.
      if (!skillBlock) {
        if (wasClosed) this.customCollapse_();
        return;
      }
      const count = parseInt(skillBlock.getFieldValue('ARGS')) || 0;
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
      const args = [];
      for (let i = 0; i < (this.paramDefs_?.length ?? 0); i++) {
        const { type } = this.paramDefs_[i];
        if (type === 'color') {
          const f = this.getField('ARG_COLOR_' + i);
          args[i] = { kind: 'color', r: f?.r_ ?? 255, g: f?.g_ ?? 0, b: f?.b_ ?? 0,
                      stale: this.staleSaved_?.has(i) ?? false };
        } else {
          const imprecise = this.impreciseSaved_?.[i];
          args[i] = { kind: 'num',
                      val: imprecise !== undefined ? imprecise : (this.getFieldValue('ARG_VAL_' + i) ?? 'NULL') };
        }
      }
      return {
        paramDefs:      this.paramDefs_ || [],
        args,
        staleSaved:     [...(this.staleSaved_ ?? [])],
        impreciseSaved: { ...(this.impreciseSaved_ ?? {}) },
      };
    },
    loadExtraState: function(state) {
      this.paramDefs_      = state.paramDefs || [];
      this.staleSaved_     = new Set(state.staleSaved || []);
      this.impreciseSaved_ = {};
      for (const [k, v] of Object.entries(state.impreciseSaved || {})) {
        this.impreciseSaved_[Number(k)] = v;
      }
      this.createParamInputsFromDefs_(this.paramDefs_);
      const args = state.args || [];
      for (let i = 0; i < this.paramDefs_.length; i++) {
        const av = args[i];
        if (!av) continue;
        const { type } = this.paramDefs_[i];
        if (av.kind === 'color' && type === 'color') {
          this.getField('ARG_COLOR_' + i)?.setRGB(av.r, av.g, av.b);
          if (av.stale) this.setFieldValue(' NULL', 'ARG_STALE_' + i);
        } else if (av.kind === 'num' && (type === 'int' || type === 'float')) {
          if (this.impreciseSaved_[i] === undefined) {
            this.getField('ARG_VAL_' + i)?.setValue(av.val);
          }
        }
      }
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

  Blockly.Blocks['condition'] = {
    init: function() {
      const getVarOptions = function() {
        const block = this.getSourceBlock();
        const opts = [['ROBOT X', 'X'], ['ROBOT Y', 'Y']];
        if (!block) return opts;
        const head = getChainHead(block);
        if (head?.type !== 'define_skill') return opts;
        const count = parseInt(head.getFieldValue('ARGS')) || 0;
        for (let i = 0; i < count; i++) {
          const name = head.getFieldValue('PARAM_NAME_' + i) || ('param' + (i + 1));
          const type = head.getFieldValue('PARAM_TYPE_' + i) || 'int';
          if (type === 'int' || type === 'float') opts.push([name, name]);
        }
        return opts;
      };
      this.appendDummyInput()
        .appendField(new FieldDropdownDark(getVarOptions), 'VAR')
        .appendField(new FieldDropdownDark([
          ['>', '>'], ['≥', '>='], ['=', '='], ['<', '<'], ['≤', '<='],
        ]), 'OP');
      this.appendValueInput('THRESHOLD').setCheck(['Number', 'Param']);
      this.setInputsInline(true);
      this.setOutput(true, 'Condition');
      this.setStyle('condition_style');
      this.setTooltip('A condition comparing a variable against a threshold. ROBOT X/Y are the robot\'s current position; parameters from the enclosing skill are also available.');
    }
  };

  Blockly.Blocks['repeat_while'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendValueInput('CONDITION').setCheck(['Condition']).appendField('Repeat while');
      this.appendStatementInput('BODY');
      this.setInputsInline(true);
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#fb923c');
      this.setTooltip('Repeat the enclosed blocks while the condition holds. Capped at 100 iterations.');
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.getInput('CONDITION')?.setVisible(false);
      if (this.getInput('SUM')) this.removeInput('SUM');
      const condBlock = this.getInput('CONDITION')?.connection?.targetBlock();
      let condText = '...';
      if (condBlock?.type === 'condition') {
        const v    = condBlock.getFieldValue('VAR') || 'X';
        const rawOp = condBlock.getFieldValue('OP') || '>';
        const opSym = { '>': '>', '>=': '>=', '=': '==', '<': '<', '<=': '<=' }[rawOp] || rawOp;
        const thresh = collapseValueInput(condBlock, 'THRESHOLD');
        const varLabel = v === 'X' ? 'ROBOT X' : v === 'Y' ? 'ROBOT Y' : v;
        condText = `${varLabel} ${opSym} ${thresh.text}`;
      }
      this.appendDummyInput('SUM').appendField(`while (${condText}):`);
      this.moveInputBefore('SUM', 'BODY');
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM')) this.removeInput('SUM');
      this.getInput('CONDITION')?.setVisible(true);
    },
  };

  Blockly.Blocks['repeat'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendValueInput('INPUT_COUNT').setCheck(['Number', 'Param']).appendField('Repeat');
      this.appendDummyInput('TIMES_LABEL').appendField('times');
      this.appendStatementInput('BODY');
      this.setInputsInline(true);
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour('#f97316');
      this.setTooltip('Repeat the enclosed blocks a number of times.');
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.getInput('INPUT_COUNT')?.setVisible(false);
      this.getInput('TIMES_LABEL')?.setVisible(false);
      // BODY stays visible so blocks can always be dropped into the loop
      if (this.getInput('SUM')) this.removeInput('SUM');
      const cnt = collapseValueInput(this, 'INPUT_COUNT');
      const row = this.appendDummyInput('SUM').appendField('for _ in range(');
      row.appendField(cnt.isParam ? new FieldLabelUnderline(cnt.text) : cnt.text);
      row.appendField('):');
      this.moveInputBefore('SUM', 'BODY');
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM')) this.removeInput('SUM');
      this.getInput('INPUT_COUNT')?.setVisible(true);
      this.getInput('TIMES_LABEL')?.setVisible(true);
    },
  };

  Blockly.Blocks['pen_up'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendDummyInput('MAIN').appendField(new Blockly.FieldLabel('Pen up'), 'LABEL');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip('Stop drawing while the robot moves.');
    },
    customCollapse_: function() { this.isCustomCollapsed_ = true;  this.setFieldValue('pen up ()',  'LABEL'); },
    customExpand_:   function() { this.isCustomCollapsed_ = false; this.setFieldValue('Pen up',     'LABEL'); },
  };

  Blockly.Blocks['pen_down'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendDummyInput('MAIN').appendField(new Blockly.FieldLabel('Pen down'), 'LABEL');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip('Resume drawing while the robot moves.');
    },
    customCollapse_: function() { this.isCustomCollapsed_ = true;  this.setFieldValue('pen down ()', 'LABEL'); },
    customExpand_:   function() { this.isCustomCollapsed_ = false; this.setFieldValue('Pen down',    'LABEL'); },
  };

  Blockly.Blocks['dip_arm'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendDummyInput('MAIN').appendField(new Blockly.FieldLabel('Dip arm in paint'), 'LABEL');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip(
        'Dip the robot arm into the nearest paint bucket to load its colour. ' +
        'Move to a bucket first, then use this block!'
      );
    },
    customCollapse_: function() { this.isCustomCollapsed_ = true;  this.setFieldValue('dip arm ()',        'LABEL'); },
    customExpand_:   function() { this.isCustomCollapsed_ = false; this.setFieldValue('Dip arm in paint', 'LABEL'); },
  };

  Blockly.Blocks['spawn_paint_bucket'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendValueInput('INPUT_X').setCheck(['Number', 'Param']).appendField('Spawn bucket at x');
      this.appendValueInput('INPUT_Y').setCheck(['Number', 'Param']).appendField('y');
      this.appendDummyInput('COLOR_ROW').appendField('color').appendField(new FieldColorSwatch(), 'COLOR');
      this.setInputsInline(true);
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip('Spawn a paint bucket at (x, y) with the chosen colour.');
    },
    customCollapse_: function() {
      this.isCustomCollapsed_ = true;
      this.getInput('INPUT_X')?.setVisible(false);
      this.getInput('INPUT_Y')?.setVisible(false);
      this.getInput('COLOR_ROW')?.setVisible(false);
      if (this.getInput('SUM')) this.removeInput('SUM');
      const x = collapseValueInput(this, 'INPUT_X');
      const y = collapseValueInput(this, 'INPUT_Y');
      const f = this.getField('COLOR');
      const r = f?.r_ ?? 255; const g = f?.g_ ?? 0; const b = f?.b_ ?? 0;
      this.appendDummyInput('SUM')
        .appendField(`spawn bucket (${x.text}, ${y.text}) rgb(${r},${g},${b})`);
    },
    customExpand_: function() {
      this.isCustomCollapsed_ = false;
      if (this.getInput('SUM')) this.removeInput('SUM');
      this.getInput('INPUT_X')?.setVisible(true);
      this.getInput('INPUT_Y')?.setVisible(true);
      this.getInput('COLOR_ROW')?.setVisible(true);
    },
  };

  Blockly.Blocks['remove_paint_bucket'] = {
    init: function() {
      this.isCustomCollapsed_ = false;
      this.appendDummyInput('MAIN').appendField(new Blockly.FieldLabel('Remove nearest bucket'), 'LABEL');
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(310);
      this.setTooltip('Remove the nearest paint bucket within reach.');
    },
    customCollapse_: function() { this.isCustomCollapsed_ = true;  this.setFieldValue('remove bucket ()', 'LABEL'); },
    customExpand_:   function() { this.isCustomCollapsed_ = false; this.setFieldValue('Remove nearest bucket', 'LABEL'); },
  };
}

export function buildToolbox(penColorEnabled = true) {
  return {
    kind: 'categoryToolbox',
    contents: [
      { kind: 'category', name: 'Program', colour: '120', contents: [
        { kind: 'block', type: 'start' },
        { kind: 'block', type: 'repeat', inputs: {
            INPUT_COUNT: { shadow: { type: 'kinder_num', fields: { NUM: 3 } } },
        }},
        { kind: 'block', type: 'repeat_while' },
        { kind: 'block', type: 'condition', inputs: {
            THRESHOLD: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
        }},
      ]},
      { kind: 'category', name: 'Movement', colour: '260', contents: [
        { kind: 'block', type: 'move_base_to_target', inputs: {
            INPUT_X: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
            INPUT_Y: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
        }},
        { kind: 'block', type: 'move_base_by', inputs: {
            INPUT_DX: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
            INPUT_DY: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
        }},
      ]},
      { kind: 'category', name: 'Pen', colour: '310', contents: [
        { kind: 'block', type: 'set_pen_color', disabled: !penColorEnabled },
        { kind: 'block', type: 'pen_down' },
        { kind: 'block', type: 'pen_up' },
        { kind: 'block', type: 'dip_arm' },
        { kind: 'block', type: 'spawn_paint_bucket', inputs: {
            INPUT_X: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
            INPUT_Y: { shadow: { type: 'kinder_num', fields: { NUM: 0 } } },
        }},
        { kind: 'block', type: 'remove_paint_bucket' },
      ]},
      { kind: 'category', name: 'Abstraction', colour: '#1e3a8a', contents: [
        { kind: 'block', type: 'define_skill' },
        { kind: 'block', type: 'use_skill' },
        { kind: 'block', type: 'param_ref' },
      ]},
    ]
  };
}

export const toolbox = buildToolbox(true);
