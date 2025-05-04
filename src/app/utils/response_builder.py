
class ResponseBuilder:
    def __init__(self):
        self.response = {
            "data": None,
            "error": None,
        }

    def setSuccess(self, success):
        self.response["success"] = success
        return self

    def setMessage(self, message):
        self.response["message"] = message
        return self
    
    def setData(self, data):
        self.response["data"] = data
        return self
    
    def setStatusCode(self, statusCode):
        self.response["statusCode"] = statusCode
        return self
    
    def setError(self, error):
        self.response["error"] = error
        return self

    def build(self):
        return self.response