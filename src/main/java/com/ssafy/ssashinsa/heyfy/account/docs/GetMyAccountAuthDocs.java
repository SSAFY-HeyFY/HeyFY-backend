package com.ssafy.ssashinsa.heyfy.account.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountAuthHttpResponseDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.ExampleObject;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

@Target(ElementType.METHOD)
@Retention(RetentionPolicy.RUNTIME)
@Operation(summary = "1-won Account Authentication", description = "Authenticates an account by transferring 1 won.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "Account authentication successful.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountAuthHttpResponseDto.class),
                        examples = @ExampleObject(
                                name = "Success Response Example",
                                value = "{\"code\": \"1234\", \"accountNo\": \"110123456789\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "Bad Request (e.g., invalid account number or authentication failed)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "Authentication Failed",
                                        summary = "Authentication number mismatch or similar failure.",
                                        value = "{\"status\": 400, \"httpError\": \"BAD_REQUEST\", \"errorCode\": \"AUTH_FAILED\", \"message\": \"Authentication failed.\"}"
                                ),
                                @ExampleObject(
                                        name = "Invalid Account Number",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"A1003\",\"message\":\"Account number is not valid.\"}"
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "Internal Server Error (e.g., external API call failed)",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "Internal Server Error",
                                value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"API_CALL_FAILED\", \"message\": \"An error has occurred.\"}"
                        )
                )
        )
})
public @interface GetMyAccountAuthDocs {
}