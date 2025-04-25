class UIPrefs {
    static instance;
    panelStates = {
        left: false,
        bottom: false,
        right: false
    };
    constructor() {
        if (UIPrefs.instance) {
            return UIPrefs.instance;
        }
        // Additional init of UIPrefs
        // localStorage.removeItem("ui_prefs");
        if (localStorage.getItem("ui_prefs")) {
            const uiPrefs = JSON.parse(localStorage.getItem("ui_prefs"));
            this.panelStates = uiPrefs.panelStates;
        }
        
        UIPrefs.instance = this;
    }

    updatePanelState(position, state) {
        this.panelStates[position] = state;
        this.storePrefs();
    }

    // Methods
    storePrefs() {
        const ps = {
            panelStates: this.panelStates
        }
        localStorage.setItem("ui_prefs", JSON.stringify(ps));
    }
}

class KnowledgeData {
    _knowledge_id;
    _domains = new Array();
    _roles = new Array();
    _categories = new Array();
    _knowledge_document;
    _document_metadata = {};
    _record_metadata
    _recall_count

    addCategory(category) {
        this._categories.push(category)
    }

    addDocumentMetadata(key, value) {
        // TODO validate the key is acceptable metadata key
        this._document_metadata[key] = value;
    }
    
    addDomain(domain) {
        // TODO validate the value with the domains list
        this._domains.push(domain);
    }

    addRole(role) {
        // TODO validate the value with the roles list
        this._roles.push(role);
    }

    set domains(domainsList) {
        // TODO Perform all the necessary reg ex checks on email string
        console.log(domainsList);
    }

    set roles(rolesList) {
        // TODO Perform all the necessary reg ex checks on email string
        console.log(rolesList);
    }

    set categories(categoriesList) {
        // TODO Perform all the necessary reg ex checks on email string
        console.log(categoriesList);
    }

    set knowledgeDocument(knowledgeDocument) {
        this._knowledge_document = knowledgeDocument;
    }
}

class KnowledgeManager {
    static instance;
    _knowledge_form;
    _tmp_knowledge_data = new KnowledgeData();

    _domain_select;
    _domain_add_btn;

    _role_select;
    _role_add_btn;

    _category_input;

    _knowledge_document;

    constructor() {
        if (KnowledgeManager.instance) {
            return KnowledgeManager.instance;
        }
        // Additional init of Knowledge Manager
        
        
        KnowledgeManager.instance = this;
    }

    // Methods

    async resetKnowledgeForm() {
        console.log("reset the knowledge form")
    }

    async submitKnowledgeForm() {
        console.log("submit the knowledge form")

        // Validate the knowledge form fields

        // Gather list values for domains, roles, and categories
    }

    validateData(data, type) {
        console.log("Validating data")
        switch (type) {
            case "domains":
                
            break;
            case "roles":
                
            break;
            case "knowledge_document":
                
            break;
        }
    }

    async addNewKnowledge(document, addedBy, domains = [], roles = [], categories = [], documentMetadata = {}) {
        const valid = this.validateData(data, type);
        if (valid) {
            const fd = new FormData();
            fd.append(name, data);
            console.log(Array.from(fd));
            
            const request = new Request(`/profile/${type}`, {
                method: "POST",
                body: fd
            });

            const response = await fetch(request);
            const responseJSON = await response.json();
            console.log(responseJSON)

            // Check the response and return results

            // update the local data storage
            this[`_${name}`] = data;
            
            return {
                success: true
            }
        }
        else {
            return {
                success: false,
                message: "Enter a valid email address"
            }
        }
    }

    async editKnowledge(knowledgeID, editedBy, changes = {}) {
        console.log("Editing knowledge")
    }

    async deleteKnowledge(knowledgeID, deletedBy) {
        console.log("Deleting a knowledge record")
    }

    async updateKnowledgeData(data, type) {
        console.log("Update the knowledge data object")
    }

    set knowledgeForm(knowledgeForm) {
        this._knowledge_form = knowledgeForm;
        // TODO Verify the form is an html form, extract all the appropriate inputs and values

        // GET domains select and add button
        this._domain_select = this._knowledge_form.elements.domain_select;
        this._domain_add_btn = this._knowledge_form.elements.domain_add;
        this._domain_add_btn.addEventListener("click", (event) => {
            const selectedValue = this._domain_select.options[this._domain_select.selectedIndex].text;
            this._tmp_knowledge_data.addDomain(selectedValue);
            console.log(this._tmp_knowledge_data);
        });
        // GET roles select and add button
        this._role_select = this._knowledge_form.elements.role_select;
        this._role_add_btn = this._knowledge_form.elements.role_add;
        this._role_add_btn.addEventListener("click", (event) => {
            const selectedValue = this._role_select.options[this._role_select.selectedIndex].text;
            this._tmp_knowledge_data.addRole(selectedValue);
            console.log(this._tmp_knowledge_data);
        });

        // GET categories select and add button
        this._category_input = this._knowledge_form.elements.category_input;
        this._category_input.addEventListener("keydown", (event) => {
            // TODO listen for enter or tab event
            if (event.key == "Enter") {
                event.preventDefault();
                const value = this._category_input.value;
                this._tmp_knowledge_data.addCategory(value);
                console.log(this._tmp_knowledge_data);
                // Clear the category input value for new value
                this._category_input.value = "";
            }
            //const selectedValue = this._category_select.options[this._category_select.selectedIndex].text;
            
        });

        // Get knowledge document - Expects a textarea wiht the name of "knowledge-document"
        this._knowledge_document = this._knowledge_form.elements.knowledge_document;
        this._knowledge_document.addEventListener("blur", (event) => {
            const documentText = event.currentTarget.value;
            this._tmp_knowledge_data.knowledgeDocument = documentText
            console.log(this._tmp_knowledge_data);
        });

        // GET knowledge metadata
        const documentTitle = this._knowledge_form.elements.document_title;
    }
}

class MemberData {
    static instance;
    _member_id;
    _member_email;
    _tg_username;
    _tg_user_id;
    _first_name;
    _last_name;
    _photo;
    _community_score;
    _register_date;
    _roles;

    constructor(basicMemberInfo = null) {
        if (MemberData.instance) {
            return MemberData.instance;
        }
        // Additional init of MemberData
        if (basicMemberInfo) {
            this._first_name = basicMemberInfo.first_name;
            this._last_name = basicMemberInfo.last_name;
            this._tg_user_id = basicMemberInfo.user_id;
            this._tg_username = basicMemberInfo.username;
            this._roles = basicMemberInfo.roles;
            this._community_score = basicMemberInfo.community_score;
            this._register_date = basicMemberInfo.register_date;
        }
        
        MemberData.instance = this;
    }

    // Methods

    validateData(data, type) {
        switch (type) {
            case "email":
                const emailRegExp = /^[\w\-\.]+@([\w-]+\.)+[\w-]{2,}$/;
                const valid = emailRegExp.test(data);

                console.log(valid);
                return valid;
            break;
        }
    }

    async storeData(data, name, type) {
        const valid = this.validateData(data, type);
        if (valid) {
            const fd = new FormData();
            fd.append(name, data);
            console.log(Array.from(fd));
            
            const request = new Request(`/profile/${type}`, {
                method: "POST",
                body: fd
            });

            const response = await fetch(request);
            const responseJSON = await response.json();
            console.log(responseJSON)

            // Check the response and return results

            // update the local data storage
            this[`_${name}`] = data;
            
            return {
                success: true
            }
        }
        else {
            return {
                success: false,
                message: "Enter a valid email address"
            }
        }
    }

    // Callback means of storing an email
    async storeEmail(inputElement) {
        const emailString = inputElement.value
        const valid = this.validateData(emailString, "email");

        if (valid) {
            const fd = new FormData();
            fd.append("member_email", emailString);
            
            const setEmailRequest = new Request("/profile/email", {
                method: "POST",
                body: fd
            });

            const response = await fetch(setEmailRequest);
            const responseJSON = await response.json()
            console.log(responseJSON)
        }

        // Update the server via an API call

        // Get response that update was successful

        // Set the value locally
        this._member_email = emailString;
    }

    // Setters
    set email(emailString) {
        // TODO Perform all the necessary reg ex checks on email string
        console.log(emailString);
    }

    
}



export { UIPrefs, MemberData, KnowledgeManager }