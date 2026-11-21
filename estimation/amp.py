"""Frozen cabinet context carried by data, checkpoints and renderers."""
import copy
import hashlib

from .chain import fingerprint

NO_AMP = 'fixed_amp_v1'


def no_amp():
    return {'amp_id':NO_AMP, 'definition':None, 'ir_sha256':None}


def validate_context(context):
    context = copy.deepcopy(context)
    if context.get('amp_id') == NO_AMP:
        if context != no_amp():
            raise ValueError('No-amp context must not contain an IR')
    else:
        definition = context.get('definition') or {}
        if (not isinstance(context.get('amp_id'),str) or not context['amp_id'] or
                context.get('amp_id') != definition.get('amp_id') or
                definition.get('render_method') != 'impulse_response' or
                not definition.get('fixed') or not definition.get('file_path') or
                not isinstance(context.get('ir_sha256'),str) or len(context['ir_sha256']) != 64 or
                any(c not in '0123456789abcdef' for c in context['ir_sha256'])):
            raise ValueError('Expected a fixed cabinet IR context')
    return context


def context_from_record(record):
    context = validate_context(record.get('amp_context',no_amp()))
    if 'amp_context' in record and record.get('amp_context_hash') != fingerprint(context):
        raise ValueError('Amp context hash mismatch')
    return context


def current_context():
    from effects.render import load_amp, find_ir
    definition = copy.deepcopy(load_amp())
    path = find_ir(definition['file_path'])
    with open(path,'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return validate_context({'amp_id':definition['amp_id'],'definition':definition,'ir_sha256':digest})


def verify_ir(context):
    context = validate_context(context)
    if context['amp_id'] != NO_AMP:
        from effects.render import find_ir
        with open(find_ir(context['definition']['file_path']),'rb') as stream:
            digest = hashlib.sha256(stream.read()).hexdigest()
        if digest != context['ir_sha256']:
            raise ValueError('Installed IR differs from the checkpoint/dataset IR')
    return context
