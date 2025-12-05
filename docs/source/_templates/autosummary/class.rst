{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :show-inheritance:

   {% block methods %}
   {% set public_methods = methods | reject("equalto", "__init__") | list %}
   {% if public_methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in public_methods %}
   {% if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
   {%- endif %}
   {%- endfor %}

   {% for item in public_methods %}
   {% if not item.startswith('_') %}
   .. automethod:: {{ item }}
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}
