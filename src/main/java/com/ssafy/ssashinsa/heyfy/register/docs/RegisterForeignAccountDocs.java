package com.ssafy.ssashinsa.heyfy.register.docs;

import com.ssafy.ssashinsa.heyfy.account.dto.AccountNoDto;
import com.ssafy.ssashinsa.heyfy.common.exception.ErrorResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
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
@Operation(summary = "Register Existing Foreign Account", description = "Verifies an existing foreign currency account via Shinhan Bank API and registers it to the service.")
@ApiResponses({
        @ApiResponse(
                responseCode = "200",
                description = "Foreign account successfully registered.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = AccountNoDto.class),
                        examples = @ExampleObject(
                                name = "Success Response Example",
                                value = "{\"accountNo\": \"0010756851096126\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "400",
                description = "Bad request or account validation error.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = {
                                @ExampleObject(
                                        name = "Account Validation Failed",
                                        summary = "Shinhan API responded that the account number is not valid.",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"A1003\",\"message\":\"Account number is not valid.\"}"
                                ),
                                @ExampleObject(
                                        name = "Account Already Exists",
                                        summary = "The provided account is already associated with this user.",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"ACCOUNT_ALREADY_EXISTS\",\"message\":\"User already has an account.\"}"
                                ),
                                @ExampleObject(
                                        name = "Account Not Matched to User",
                                        summary = "The provided account does not belong to the authenticated user.",
                                        value = "{\"status\":400,\"httpError\":\"BAD_REQUEST\",\"errorCode\":\"ACCOUNT_NOT_MATCH\",\"message\":\"Account number does not match.\"}"
                                )
                        }
                )
        ),
        @ApiResponse(
                responseCode = "404",
                description = "User not found.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "User Not Found",
                                value = "{\"status\": 404, \"httpError\": \"NOT_FOUND\", \"errorCode\": \"USER_NOT_FOUND\", \"message\": \"User not found.\"}"
                        )
                )
        ),
        @ApiResponse(
                responseCode = "500",
                description = "Internal Server Error or Shinhan API call failed.",
                content = @Content(
                        mediaType = "application/json",
                        schema = @Schema(implementation = ErrorResponse.class),
                        examples = @ExampleObject(
                                name = "API Call Failed",
                                value = "{\"status\": 500, \"httpError\": \"INTERNAL_SERVER_ERROR\", \"errorCode\": \"API_CALL_FAILED\", \"message\": \"Shinhan API call failed.\"}"
                        )
                )
        )
})
@Parameter(name = "Authorization", description = "JWT 액세스 토큰 (Bearer <token>)", in = ParameterIn.HEADER)
@Parameter(name = "sid", description = "세션 ID", in = ParameterIn.HEADER)
public @interface RegisterForeignAccountDocs {
}