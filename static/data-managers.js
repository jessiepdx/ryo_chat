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


export { UIPrefs, MemberData }