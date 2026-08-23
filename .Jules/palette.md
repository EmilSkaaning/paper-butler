## 2026-08-09 - [Streamlit UX: Label Collapse with Placeholders]
**Learning:** Streamlit input widgets (like `st.multiselect` and `st_keyup`) can consume significant vertical real estate when their labels are visible. However, removing labels entirely breaks screen reader accessibility. By using `label_visibility="collapsed"` combined with descriptive `placeholder` arguments (e.g., `placeholder="Filter by status..."`), we can achieve a denser, cleaner UI while still preserving the structural label element required by screen readers and satisfying the "good UX is invisible" philosophy.
**Action:** When space is tight, explicitly hide labels using `label_visibility="collapsed"` rather than omitting the `label` prop or popping it from session state, and ALWAYS provide a helpful `placeholder` string so visual users still understand the widget's purpose.

## 2026-08-15 - [Streamlit UX: Dynamic Tooltips for Disabled Buttons]
**Learning:** Streamlit allows dynamic strings for `help` parameters on disabled buttons. When an icon-only button is disabled, showing a static tooltip (like "Delete selected papers") is less helpful than explaining *why* it's disabled or what action the user needs to take (e.g., "Select papers to delete"). This makes the empty/disabled state actionable and improves usability.
**Action:** Always use conditional expressions for tooltips on disabled interactive elements to explain the required precondition rather than just stating the element's purpose.

## 2026-08-23 - [Streamlit UX: Compound Disabled States and Tooltips]
**Learning:** When buttons have multiple independent disabled conditions (like requiring both an API token and a valid selection/file state), a simple binary `if/else` tooltip might mask the primary blocking reason. Users who satisfy one condition might still see the generic tooltip without realizing another requirement is missing.
**Action:** When an element is disabled for multiple potential reasons, dynamically cascade the `help` parameter using an `if/elif/else` structure to specifically identify the most relevant missing precondition (e.g., first missing token, then missing selection).
