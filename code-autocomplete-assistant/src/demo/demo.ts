interface UserProfile {
    id: string;
    name: string;
    email: string;
    avatarUrl: string;
    createdAt: string;
    lastSeenAt: string; // TODO: add type
    lastLocation: string; // TODO: add type
    location: string; // TODO: add type
    status: string; // TODO: add type
    isOnline:
        | boolean
        | {
              status: string;
              lastSeenAt: string;
          }; // TODO: add type
    isTyping: boolean; // TODO: add type
    isBlocked: boolean; // TODO: add type
    isMuted: boolean; // TODO: add
}

// Function to take user input
function() {
			var oModel = this._oDialog.getModel("view"),
				oDialog = this._oDialog,
				oDialogModel = oDialog.getModel("view"),
				oDialogContent = oDialog.getContent()[0],
				oDialogContentModel = oDialogContent.getModel("view"),
				oDialogContentModelClone = oDialogContentModel.clone(),
				oDialogContentModelClone = oDialogContentModelClone.bind(this);

			oDialogContentModelClone.setProperty("/name", oModel.getProperty("/name"));
			oDialogContentModelClone.setProperty("/email", oModel.getProperty("/email"));
			oDialogContentModelClone.setProperty("/phone", oModel.getProperty("/phone"));
			oDialogContentModelClone.setProperty("/message", oModel.getProperty("/message"));

			oDialogModel.setProperty("/name", oModel.getProperty("/name"));
			oDialogModel.setProperty("/email", oModel.getProperty("/email"));
			oDialogModel.setProperty("/phone", oModel.getProperty("/phone"));
			oDialogModel.setProperty("/message", oModel.getProperty("/message"));

			oDialog.open();
		}